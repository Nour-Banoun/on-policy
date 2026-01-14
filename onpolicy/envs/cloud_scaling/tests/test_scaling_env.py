import numpy as np
import pytest

from onpolicy.envs.cloud_scaling.scaling_env import CloudScalingEnv
# pytest -q onpolicy/envs/cloud_scaling/tests/test_scaling_env.py::test_reset_shapes_and_mask -q

def test_reset_shapes_and_mask():
    env = CloudScalingEnv(all_args=None, max_agents=4, init_agents=2, obs_dim=8, episode_length=10)
    obs, share_obs, avail = env.reset()

    assert obs.shape == (env.max_agents, env.obs_dim)
    assert share_obs.shape == (env.max_agents, env.share_obs_dim)
    assert avail.shape == (env.max_agents, 3)

    # active mask has correct number of active slots
    assert int(np.sum(env.active_mask)) == env.init_agents

    # inactive slots should have zero observations
    for i in range(env.max_agents):
        if not env.active_mask[i]:
            assert np.all(obs[i] == 0.0)


def test_step_prev_action_effects_and_reward_shape():
    env = CloudScalingEnv(all_args=None, max_agents=3, init_agents=2, obs_dim=8, episode_length=10)
    obs, share_obs, avail = env.reset()

    # initial running instances for active slots should be 2
    active_indices = [i for i, a in enumerate(env.active_mask) if a]
    for i in active_indices:
        assert env.instance_count[i] == 2

    # set previous action to scale_out (1) for slot 0 so step applies it
    env.last_action[0] = 1

    # perform a step (incoming actions are hold=1 for all slots)
    actions = [1] * env.max_agents
    obs2, share2, rewards, dones, infos, avail2 = env.step(actions)

    # instance_count for slot 0 should have increased by at least 1
    assert env.instance_count[0] >= 3

    # rewards is a list of length max_agents with one-element lists inside
    assert isinstance(rewards, list) and len(rewards) == env.max_agents
    for r in rewards:
        assert isinstance(r, list) and len(r) == 1
        assert isinstance(r[0], float)

    # infos for active slot contains expected keys
    info0 = infos[0]
    assert 'current_cost' in info0
    assert 'instance_running' in info0

    # dones is a list of booleans length max_agents
    assert isinstance(dones, list) and len(dones) == env.max_agents
    for d in dones:
        assert isinstance(d, bool)
