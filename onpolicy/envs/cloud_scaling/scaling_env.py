import numpy as np
from gym import spaces

from .simulator import SimpleSimulator

import os
import json
from pathlib import Path

# Cache defaults (user-local cache)
CACHE_PATH = Path(os.path.expanduser("~")) / ".onpolicy_cloud_scaling_cache.json"
DEFAULT_CACHE = {
    "Number_of_agents": 1,
    "cost_per_agent_per_hour": 0.01,
    "cost_threshold_per_month": 10.0
}

"""
CloudScalingEnv

Multi-agent environment skeleton for dynamic VM autoscaling.
- dynamic agent membership using a fixed `max_agents` slot pool
- discrete 3-way actions per agent: 0 -> scale_in (-1), 1 -> hold (0), 2 -> scale_out (+1)
- observations per agent: vector of normalized metrics
- returns: obs (list of per-agent arrays), share_obs (list of identical global obs arrays),
  rewards (list of [reward] per agent), dones (list per agent), infos (list per agent),
  available_actions (np.ndarray per agent)
"""

class CloudScalingEnv(object):
    def __init__(self, all_args=None, max_agents=8, init_agents=3, obs_dim=8, episode_length=200):
        # configuration
        self.all_args = all_args
        self.max_agents = int(max_agents)
        self.init_agents = max(1, int(init_agents))
        self.obs_dim = int(obs_dim)
        self.episode_length = int(episode_length)
        self.current_step = 0

        # per-step duration (seconds). read from args if present, default 60s
        self.step_seconds = int(getattr(all_args, "step_seconds", 60)) if all_args is not None else 60

        # load or initialize cache early so we can use cached values right away
        self.CACHE_PATH = CACHE_PATH
        self.DEFAULT_CACHE = DEFAULT_CACHE
        self.cache = self._load_cache()

        # allow constructor or all_args to override the cache-initialized defaults
        # priority: explicit constructor arg > all_args > cache > hard default
        if all_args is not None and hasattr(all_args, "Number_of_agents"):
            self.cache["Number_of_agents"] = int(getattr(all_args, "Number_of_agents"))
        if init_agents is not None:
            # if user passed init_agents explicitly, respect it
            self.cache["Number_of_agents"] = int(init_agents)

        if all_args is not None and hasattr(all_args, "cost_per_agent_per_hour"):
            self.cache["cost_per_agent_per_hour"] = float(getattr(all_args, "cost_per_agent_per_hour"))

        if all_args is not None and hasattr(all_args, "cost_threshold_per_month"):
            self.cache["cost_threshold_per_month"] = float(getattr(all_args, "cost_threshold_per_month"))

        # persist any changes back to disk
        self._save_cache()

        # set derived values: use cached Number_of_agents as initial active agents
        self.init_agents = max(1, int(self.cache.get("Number_of_agents", self.init_agents)))

        # action / observation spaces (one entry per slot up to max_agents)
        # action: 3 discrete options per agent
        self.action_space = [spaces.Discrete(3) for _ in range(self.max_agents)]
        # per-agent observation (cpu, mem, net, throughput, latency, instance_norm, last_action_onehot (3->scaled), time_since_last_scale_norm)
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
                                  for _ in range(self.max_agents)]

        # shared observation is global concat of per-agent obs (fixed length)
        self.share_obs_dim = self.obs_dim * self.max_agents
        self.share_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,),
                                                  dtype=np.float32) for _ in range(self.max_agents)]

        # bookkeeping per slot
        self.active_mask = np.zeros(self.max_agents, dtype=bool)
        self.instance_count = np.zeros(self.max_agents, dtype=np.int32)  # 1 if active, 0 if inactive
        self.last_action = np.zeros(self.max_agents, dtype=np.int32)  # -1/0/+1 stored as -1,0,1

        # simulator (simple workload & latency model) - pass step_seconds if simulator needs it
        try:
            self.sim = SimpleSimulator(self.max_agents, step_seconds=self.step_seconds)
        except TypeError:
            # older simulator signature: ignore step_seconds
            self.sim = SimpleSimulator(self.max_agents)

        # seed if available
        if all_args is not None and hasattr(all_args, "seed"):
            self.seed(getattr(all_args, "seed"))

        # initialize env to default active set
        self._init_env()

        self.max_create_per_step = int(getattr(all_args, "max_create_per_step", 1))  # default 1 new slot per step

    def _init_env(self):
        # activate first init_agents slots
        self.active_mask[:] = False
        for i in range(min(self.init_agents, self.max_agents)):
            self.active_mask[i] = True
            self.instance_count[i] = 1  # each active slot represents one VM/instance
            self.last_action[i] = 0
        self.current_step = 0
        self.sim.reset()
        self._update_obs_cache()

    def seed(self, seed=None):
        if seed is None:
            seed = 1
        np.random.seed(seed)
        self.sim.seed(seed)

    def _obs_for_slot(self, slot_idx):
        """
        Build observation vector for a slot (active or inactive).
        For inactive slots we return zeros.
        """
        if not self.active_mask[slot_idx]:
            return np.zeros(self.obs_dim, dtype=np.float32)

        # fetch metrics from simulator (simulator stores last_metrics)
        metrics = self.sim.get_metrics(slot_idx)
        # metrics dict keys: cpu, mem, net, throughput, latency
        cpu = float(metrics.get("cpu", 0.0))
        mem = float(metrics.get("mem", 0.0))
        net = float(metrics.get("net", 0.0))
        throughput = float(metrics.get("throughput", 0.0))
        latency = float(metrics.get("latency", 0.0))

        instance_norm = float(self.instance_count[slot_idx]) / max(1.0, np.max(self.instance_count))
        last_action_val = float(self.last_action[slot_idx])  # -1,0,1

        time_since_scale_norm = float(self.sim.time_since_last_scale(slot_idx)) / max(1.0, self.episode_length)

        # observation vector layout:
        obs = np.array([
            cpu,
            mem,
            net,
            throughput,
            np.tanh(latency / 1000.0),  # soft normalize latency
            instance_norm,
            (last_action_val + 1.0) / 2.0,  # map -1/0/1 to 0..1
            time_since_scale_norm
        ], dtype=np.float32)

        # pad/truncate to obs_dim
        if self.obs_dim > obs.shape[0]:
            pad = np.zeros(self.obs_dim - obs.shape[0], dtype=np.float32)
            obs = np.concatenate([obs, pad])
        elif self.obs_dim < obs.shape[0]:
            obs = obs[:self.obs_dim]

        return obs

    def _update_obs_cache(self):
        # recompute cached obs arrays and shared observation
        self._obs_cache = [self._obs_for_slot(i) for i in range(self.max_agents)]
        # shared obs: concat of per-slot obs
        flat = np.concatenate(self._obs_cache).astype(np.float32)
        self._share_obs_cache = [flat.copy() for _ in range(self.max_agents)]
        # available actions: ones for active, zeros for inactive (shape act_dim)
        self._avail_actions_cache = np.ones((self.max_agents, 3), dtype=np.float32)
        for i in range(self.max_agents):
            if not self.active_mask[i]:
                self._avail_actions_cache[i] = np.zeros(3, dtype=np.float32)

    def reset(self, choose=None):
        """
        Reset environment.
        Returns (obs, share_obs, available_actions)
        - obs: np.ndarray shaped (max_agents, obs_dim)
        - share_obs: np.ndarray shaped (max_agents, share_obs_dim)
        - available_actions: np.ndarray shaped (max_agents, act_dim)
        """
        self._init_env()
        self._update_obs_cache()
        obs = np.stack(self._obs_cache).astype(np.float32)
        share_obs = np.stack(self._share_obs_cache).astype(np.float32)
        available_actions = self._avail_actions_cache.copy()
        return obs , share_obs, available_actions
    """return obs, share_obs, available_actions"""

    def step(self, actions):
        """
        Step the environment.
        Returns:
        obs, share_obs, rewards, dones, infos, available_actions
        """
        self.current_step += 1

        # sanitize actions
        a = np.array(actions).reshape(-1)
        if a.shape[0] < self.max_agents:
            # pad with hold (1)
            pad = np.ones(self.max_agents - a.shape[0], dtype=int)
            a = np.concatenate([a, pad])
        elif a.shape[0] > self.max_agents:
            a = a[:self.max_agents]

        # translate discrete actions: 0 -> -1 (scale_in), 1 -> 0 (hold), 2 -> +1 (scale_out)
        delta = np.zeros(self.max_agents, dtype=int)
        delta[a == 0] = -1
        delta[a == 1] = 0
        delta[a == 2] = +1

        # process removals
        remove_requests = np.where((delta == -1) & (self.active_mask))[0]
        for idx in remove_requests:
            self.active_mask[idx] = False
            self.instance_count[idx] = 0
            self.last_action[idx] = -1
            self.sim.mark_removed(idx)

        # process creates: count +1 requests from active agents and instantiate in free slots
        requested_creates = int(np.sum((delta == 1) & (self.active_mask)))
        if requested_creates > 0:
            free_slots = np.where(~self.active_mask)[0]
            # cap number created per step
            num_to_create = min(requested_creates, len(free_slots), self.max_create_per_step)
            to_fill = free_slots[:num_to_create]
            for s in to_fill:
                self.active_mask[s] = True
                self.instance_count[s] = 1
                self.last_action[s] = 1
                self.sim.mark_created(s)

        # holds
        for i in range(self.max_agents):
            if self.active_mask[i] and delta[i] == 0:
                self.last_action[i] = 0

        # simulator step
        self.sim.step(self.active_mask.copy(), self.instance_count.copy())

        # compute rewards per agent
        rewards = []
        infos = []
        for i in range(self.max_agents):
            if not self.active_mask[i]:
                rewards.append([0.0])
                infos.append({"active": False})
                continue

            metrics = self.sim.get_metrics(i)
            cpu = metrics.get("cpu", 0.0)
            latency = metrics.get("latency", 0.0)
            throughput = metrics.get("throughput", 0.0)

            # cost per step derived from per-hour cost stored in cache
            cost_per_hour = float(self.cache.get("cost_per_agent_per_hour", 0.01))
            cost_per_step = cost_per_hour * (self.step_seconds / 3600.0)
            cost_per_agent = cost_per_step

            sla_threshold_ms = 200.0
            sla_penalty = 100.0
            scale_penalty = 0.05 * abs(self.last_action[i])

            reward = - cost_per_agent * float(self.instance_count[i])
            if latency > sla_threshold_ms:
                reward -= sla_penalty * (latency - sla_threshold_ms) / sla_threshold_ms
            reward += 0.05 * min(throughput / max(metrics.get("cap", 1.0), 1.0), 1.0)
            reward -= scale_penalty

            rewards.append([float(reward)])
            infos.append({
                "active": True,
                "cpu": cpu,
                "latency": latency,
                "throughput": throughput
            })

        # done flags: end episode when current_step >= episode_length
        done_flag = (self.current_step >= self.episode_length)
        dones = [bool(done_flag) for _ in range(self.max_agents)]

        # update cached observations
        self._update_obs_cache()
        obs = np.stack(self._obs_cache).astype(np.float32)
        share_obs = np.stack(self._share_obs_cache).astype(np.float32)
        available_actions = self._avail_actions_cache.copy()

        return obs, share_obs, rewards, dones, infos, available_actions

    def render(self, mode='human'):
        info = {
            "step": self.current_step,
            "active": int(np.sum(self.active_mask)),
            "instances": self.instance_count.tolist()
        }
        if mode == 'human':
            print("CloudScalingEnv:", info)
        return info

    def _load_cache(self):
        try:
            if self.CACHE_PATH.exists():
                with open(self.CACHE_PATH, "r") as f:
                    data = json.load(f)
                # ensure default keys exist
                for k, v in self.DEFAULT_CACHE.items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception:
            pass
        return dict(self.DEFAULT_CACHE)

    def _save_cache(self):
        try:
            with open(self.CACHE_PATH, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass

    def update_cache(self, key, value, persist=True):
        """
        Update cache value and optionally persist to disk.
        Example: env.update_cache('Number_of_agents', 3)
        """
        self.cache[key] = value
        if persist:
            self._save_cache()

    def get_active_mask(self):
        """
        Return a copy of the boolean active mask (length == max_agents).
        Trainers/runners can call `env.get_active_mask()` or read `env.active_mask`.
        """
        return self.active_mask.copy()