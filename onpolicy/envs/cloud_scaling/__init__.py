from .scaling_env import CloudScalingEnv  # primary env class

# Backwards-compatible alias: some code expects `ScalingEnv`
ScalingEnv = CloudScalingEnv

__all__ = ["CloudScalingEnv", "ScalingEnv"]