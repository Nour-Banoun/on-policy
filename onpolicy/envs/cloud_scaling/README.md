CloudScalingEnv (skeleton)

- Location: onpolicy/envs/cloud_scaling/
- Purpose: Multi-agent environment skeleton modeling dynamic VM autoscaling.
- Key files:
  - scaling_env.py: environment class + CloudScalingEnv(all_args) factory.
  - simulator.py: simple synthetic workload and latency model.
- Observation per agent (default 8 dims): [cpu, mem, net, throughput, latency_scaled, instance_norm, last_action_scaled, time_since_scale_norm]
- Action per agent: Discrete(3) -> {0: scale_in, 1: hold, 2: scale_out}
- Dynamic agents: environment manages a fixed slot pool `max_agents`; active slots are indicated by a boolean `active_mask`.
- Integration: training scripts can call `from onpolicy.envs.cloud_scaling.scaling_env import CloudScalingEnv` and instantiate via `env = CloudScalingEnv(all_args, max_agents=8, init_agents=3)`.

Cache file: ~/.onpolicy_cloud_scaling_cache.json
Default values:
  - Number_of_agents: 1
  - cost_per_agent_per_hour: 0.01
  - cost_threshold_per_month: 10.0
You can override these with CLI args: --Number_of_agents --cost_per_agent_per_hour --cost_threshold_per_month