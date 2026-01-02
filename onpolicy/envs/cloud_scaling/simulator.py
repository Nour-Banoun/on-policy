import numpy as np
import math
"""
SimpleSimulator

Very small synthetic workload and latency model for the CloudScalingEnv.
- Provides per-slot metrics: cpu, mem, net, throughput, latency
- Exposes .step(active_mask, instance_count) and .get_metrics(slot)
- Exposes helper functions mark_created/mark_removed/time_since_last_scale
"""

class SimpleSimulator:
    def __init__(self, max_slots=8, base_capacity_per_instance=100.0, seed=0, step_seconds=60):
        self.max_slots = int(max_slots)
        self.base_capacity = float(base_capacity_per_instance)
        self._rng = np.random.RandomState(seed if seed is not None else 0)
        self.t = 0
        self.step_seconds = int(step_seconds)
        # per-slot metrics
        self.metrics = [self._empty_metrics() for _ in range(self.max_slots)]
        # per-slot last scale time
        self.last_scale_time = [-999 for _ in range(self.max_slots)]

    def _empty_metrics(self):
        return {"cpu": 0.0, "mem": 0.0, "net": 0.0, "throughput": 0.0, "latency": 0.0, "cap": self.base_capacity}

    def seed(self, s):
        self._rng = np.random.RandomState(s if s is not None else 0)

    def reset(self):
        self.t = 0
        self.metrics = [self._empty_metrics() for _ in range(self.max_slots)]
        self.last_scale_time = [-999 for _ in range(self.max_slots)]

    def mark_created(self, slot):
        self.last_scale_time[slot] = self.t

    def mark_removed(self, slot):
        self.last_scale_time[slot] = self.t

    def time_since_last_scale(self, slot):
        return max(0, self.t - self.last_scale_time[slot])

    def step(self, active_mask, instance_counts):
        """
        active_mask: boolean array length max_slots
        instance_counts: int array length max_slots (1 or 0)
        We'll produce a synthetic, time-varying load distributed across active slots.
        """
        self.t += 1

        # global workload: sinusoid + random spikes
        base = 50.0 + 40.0 * (0.5 + 0.5 * math.sin(self.t / 24.0))
        spike = 0.0
        if self._rng.rand() < 0.05:
            spike = self._rng.rand() * 200.0

        global_load = base + spike

        # distribute load across active slots (equal sharing)
        active_indices = [i for i, a in enumerate(active_mask) if a]
        num_active = len(active_indices)
        if num_active == 0:
            per_slot_load = 0.0
            cap = self.base_capacity
        else:
            per_slot_load = global_load / max(1, num_active)
            cap = self.base_capacity

        for i in range(self.max_slots):
            if not active_mask[i]:
                self.metrics[i] = self._empty_metrics()
            else:
                # compute utilization
                util = min(1.0, per_slot_load / (cap + 1e-8))
                cpu = util * (0.8 + 0.2 * self._rng.rand())
                mem = min(1.0, 0.5 + 0.5 * util + 0.1 * self._rng.rand())
                net = min(1.0, 0.4 + 0.6 * util + 0.1 * self._rng.rand())
                throughput = per_slot_load * (0.8 + 0.4 * (1.0 - util * 0.5))  # heuristic
                # latency model: small when util < 0.7, grows quickly beyond
                if util < 0.7:
                    latency = 50.0 + 50.0 * util + 10.0 * self._rng.rand()
                else:
                    latency = 50.0 + 300.0 * (util - 0.7) / 0.3 + 50.0 * self._rng.rand()
                self.metrics[i] = {
                    "cpu": float(np.clip(cpu, 0.0, 1.0)),
                    "mem": float(np.clip(mem, 0.0, 1.0)),
                    "net": float(np.clip(net, 0.0, 1.0)),
                    "throughput": float(max(0.0, throughput)),
                    "latency": float(max(0.0, latency)),
                    "cap": float(cap)
                }

    def get_metrics(self, slot_idx):
        return self.metrics[int(slot_idx)]