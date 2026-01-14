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

        # convenience aliases / runtime defaults exposed on the env
        # legacy cost fields (kept for backward compatibility)
        self.cost_per_agent_per_hour = float(self.cache.get("cost_per_agent_per_hour", 0.01))
        self.cost_limitation_month = float(self.cache.get("cost_threshold_per_month", 10.0))
        # new experimental parameters
        # per-instance cost options (user requested: 5:5:50)
        self.instance_costs = np.arange(5, 51, 5).astype(np.float32)
        # per-month cost limitation options (50:50:1000)
        self.cost_limitations = np.arange(50, 1001, 50).astype(np.float32)
        # selected indices (defaults to first value)
        self.instance_cost_idx = 0
        self.cost_limitation_idx = 0
        # current scalar values (derived)
        self.instance_cost = float(self.instance_costs[self.instance_cost_idx])
        self.cost_limitation = float(self.cost_limitations[self.cost_limitation_idx])
        # alias for clarity: maximum instances per slot (static default requested = 1000)
        self.max_instances = int(getattr(all_args, "max_instances", 1000))
        # tunable weight for utilization penalty in reward (can be passed via CLI/all_args)
        self.util_penalty_weight = float(getattr(all_args, "util_penalty_weight", 100.0))

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
            # initial running instances per active slot (user requested initial value = 2)
            self.instance_count[i] = 2  # number of instances running in this slot
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

        # report raw running instance count (0 or 1) rather than a normalized value
        instance_running = float(self.instance_count[slot_idx])
        last_action_val = float(self.last_action[slot_idx])  # -1,0,1

        time_since_scale_norm = float(self.sim.time_since_last_scale(slot_idx)) / max(1.0, self.episode_length)

        # observation vector layout:
        obs = np.array([
            cpu,
            mem,
            net,
            throughput,
            np.tanh(latency / 1000.0),  # soft normalize latency
            instance_running,
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

        # sanitize actions (these are the new actions that will take effect next step)
        a = np.array(actions).reshape(-1)
        if a.shape[0] < self.max_agents:
            pad = np.ones(self.max_agents - a.shape[0], dtype=int)  # default hold
            a = np.concatenate([a, pad])
        elif a.shape[0] > self.max_agents:
            a = a[:self.max_agents]

        # map incoming actions to stored last_action values for next step
        incoming_delta = np.zeros(self.max_agents, dtype=int)
        incoming_delta[a == 0] = -1
        incoming_delta[a == 1] = 0
        incoming_delta[a == 2] = 1

        # Apply effects of the previous actions (self.last_action) now.
        # Reward for the previous action is computed using metrics after these updates.
        prev_action = self.last_action.copy()

        # Update instance counts based on previous action (scale out/in affect instance_running)
        for i in range(self.max_agents):
            if not self.active_mask[i]:
                continue
            if prev_action[i] == 1:
                # previous step requested scale out -> increase running instances
                self.instance_count[i] = min(self.max_instances, int(self.instance_count[i]) + 1)
            elif prev_action[i] == -1:
                # previous step requested scale in -> decrease, but not below 1
                self.instance_count[i] = max(1, int(self.instance_count[i]) - 1)

        # update simulator with current instance counts
        self.sim.step(self.active_mask.copy(), self.instance_count.copy())

        # compute rewards for previous actions (using updated metrics)
        rewards = []
        infos = []
        for i in range(self.max_agents):
            if not self.active_mask[i]:
                rewards.append([0.0])
                infos.append({
                    "active": False,
                    "current_cost": 0.0,
                    "cost_limitation": float(self.cost_limitation),
                    "max_instances": int(self.max_instances)
                })
                continue

            metrics = self.sim.get_metrics(i)
            cpu = float(metrics.get("cpu", 0.0))
            mem = float(metrics.get("mem", 0.0))
            net = float(metrics.get("net", 0.0))
            throughput = float(metrics.get("throughput", 0.0))
            latency = float(metrics.get("latency", 0.0))
            cap = float(metrics.get("cap", self.sim.base_capacity))

            # determine per-slot instance cost (env-level scalar) and current_cost
            self.instance_cost = float(self.instance_costs[self.instance_cost_idx])
            self.cost_limitation = float(self.cost_limitations[self.cost_limitation_idx])
            current_cost = float(self.instance_count[i]) * float(self.instance_cost)

            # Apply multiplier to metrics based on the PREVIOUS action
            if prev_action[i] == 1:
                # scale out: factor = (old_count)/(new_count) = (instance_count-1)/instance_count
                if self.instance_count[i] > 0:
                    factor = float(self.instance_count[i] - 1) / float(self.instance_count[i])
                else:
                    factor = 1.0
                cpu *= factor
                mem *= factor
                net *= factor
                throughput *= factor
                latency *= factor
                cap *= factor
            elif prev_action[i] == -1:
                # scale in: factor = (old_count)/(new_count) = (instance_count+1)/instance_count
                factor = float(self.instance_count[i] + 1) / float(self.instance_count[i]) if self.instance_count[i] > 0 else 1.0
                cpu *= factor
                mem *= factor
                net *= factor
                throughput *= factor
                latency *= factor
                cap *= factor
            else:
                # no-scale: leave metrics as produced (randomized by simulator)
                pass

            # reward computed for the PREVIOUS action using updated metrics
            # Base reward now encourages staying under the monthly cost limitation
            sla_threshold_ms = 200.0
            sla_penalty = 100.0
            scale_penalty = 0.05 * abs(prev_action[i])

            # Base reward: positive when current cost is below the cost limitation
            cost_penalty_weight = 200.0
            reward = cost_penalty_weight * float(self.cost_limitation - current_cost)/self.cost_limitation

            # Consider cpu, mem, net utilizations (values in 0..1 from simulator)
            # We penalize higher utilization to encourage sufficient capacity.
            util_avg = (cpu + mem + net) / 3.0
            # Tunable weight (can be passed via CLI as --util_penalty_weight)
            reward -= float(self.util_penalty_weight) * util_avg

            # SLA latency penalty (unchanged): heavy penalty when latency exceeds threshold
            if latency > sla_threshold_ms:
                reward -= sla_penalty * (latency - sla_threshold_ms) / sla_threshold_ms

            # Throughput bonus (small) normalized by capacity
            reward += 0.05 * min(throughput / max(cap, 1.0), 1.0)

            # Small penalty for scaling actions (encourage stability)
            reward -= scale_penalty

            rewards.append([float(reward)])
            infos.append({
                "active": True,
                "cpu": float(cpu),
                "mem": float(mem),
                "net": float(net),
                "throughput": float(throughput),
                "latency": float(latency),
                "cap": float(cap),
                "current_cost": float(current_cost),
                "instance_running": int(self.instance_count[i]),
                "cost_limitation": float(self.cost_limitation),
                "max_instances": int(self.max_instances)
            })

        # finally, store incoming actions as last_action for use in next step
        self.last_action = incoming_delta.copy()

        # done flags: end episode when current_step >= episode_length
        done_flag = (self.current_step >= self.episode_length)
        dones = [bool(done_flag) for _ in range(self.max_agents)]

        # update cached observations
        self._update_obs_cache()
        obs = np.stack(self._obs_cache).astype(np.float32)
        share_obs = np.stack(self._share_obs_cache).astype(np.float32)
        available_actions = self._avail_actions_cache.copy()

        return obs, share_obs, rewards, dones, infos, available_actions
    
    # def close(self):
    #     """
    #     Clean up resources used by the environment.
    #     VecEnv wrappers call `env.close()` during teardown, so provide
    #     a safe implementation that tolerates missing simulator close.
    #     """
    #     try:
    #         if hasattr(self, "sim") and hasattr(self.sim, "close"):
    #             try:
    #                 self.sim.close()
    #             except Exception:
    #                 pass
    #     except Exception:
    #         pass

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

    def set_instance_cost_index(self, idx, reset_running=True):
        """Set the index into `instance_costs`. Optionally reset running instances to initial value."""
        idx = int(idx) % len(self.instance_costs)
        self.instance_cost_idx = idx
        self.instance_cost = float(self.instance_costs[self.instance_cost_idx])
        if reset_running:
            # reset running instances for active slots to initial value (2)
            for i in range(self.max_agents):
                if self.active_mask[i]:
                    self.instance_count[i] = 2

    def set_cost_limitation_index(self, idx, reset_running=True):
        """Set the index into `cost_limitations`. Optionally reset running instances to initial value."""
        idx = int(idx) % len(self.cost_limitations)
        self.cost_limitation_idx = idx
        self.cost_limitation = float(self.cost_limitations[self.cost_limitation_idx])
        if reset_running:
            for i in range(self.max_agents):
                if self.active_mask[i]:
                    self.instance_count[i] = 2

    def get_active_mask(self):
        """
        Return a copy of the boolean active mask (length == max_agents).
        Trainers/runners can call `env.get_active_mask()` or read `env.active_mask`.
        """
        return self.active_mask.copy()

    def close(self):
        """
        Clean up resources used by the environment.
        VecEnv wrappers call `env.close()` during teardown, so provide
        a safe implementation that tolerates missing simulator close.
        """
        try:
            if hasattr(self, "sim") and hasattr(self.sim, "close"):
                try:
                    self.sim.close()
                except Exception:
                    pass
        except Exception:
            pass