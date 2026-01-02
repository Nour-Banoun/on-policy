import socket
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])

# expose cloud scaling env factory for convenience
try:
    from .cloud_scaling.scaling_env import CloudScalingEnv  # noqa: F401
except Exception:
    # import may fail if file not yet created or dependencies missing; swallow to avoid breaking other imports
    CloudScalingEnv = None


