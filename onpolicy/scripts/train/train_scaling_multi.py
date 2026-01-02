#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import make_scaling_train_env, make_scaling_eval_env
from onpolicy.envs.cloud_scaling.scaling_env import CloudScalingEnv
from onpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
"""Train script for the CloudScaling multi-agent env (recurrent MAPPO by default)."""

def main(args):
    parser = get_config()
    # parse known args using the shared config parser (avoid re-adding already-defined flags)
    all_args = parser.parse_known_args(args)[0]

    # set defaults for cloud-scaling-specific params if not provided on CLI
    if not hasattr(all_args, "max_agents"):
        all_args.max_agents = 8
    if not hasattr(all_args, "init_agents"):
        all_args.init_agents = 1
    if not hasattr(all_args, "obs_dim"):
        all_args.obs_dim = 8
    if not hasattr(all_args, "episode_length"):
        all_args.episode_length = 200
    # ensure runner-expected metadata exists (used for logging)
    if not hasattr(all_args, "scenario_name"):
        all_args.scenario_name = getattr(all_args, "env_name", "cloud_scaling")
    if not hasattr(all_args, "experiment_name"):
        all_args.experiment_name = getattr(all_args, "experiment_name", "default")
    if not hasattr(all_args, "user_name"):
        all_args.user_name = getattr(all_args, "user_name", "user")

    # set defaults for algorithm flags
    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappo".upper():
        print("Using R-MAPPO; enabling recurrent policy")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("Using MAPPO; disabling recurrent policy by default")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("Using IPPO; disabling centralized value")
        all_args.use_centralized_V = False
    else:
        # If unknown algorithm provided, leave as-is — trainer will raise later as needed
        pass

    # device: GPU if available and requested
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # choose a run number directory
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init using the compatibility factory (returns classic 4-tuple API)
    envs = make_scaling_train_env(all_args)
    eval_envs = make_scaling_eval_env(all_args) if all_args.use_eval else None
    num_agents = getattr(all_args, "max_agents", 8)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # runner selection based on whether policies are shared
    if all_args.share_policy:
        from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
        
    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])