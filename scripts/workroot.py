"""Locations used by the analysis scripts that have no --env argument of their own.

    ORGANIC3D_ENV                env config (default: configs/env/local.yaml, copied from configs/env/TEMPLATE.yaml)
    ORGANIC3D_WORK_ROOT          overrides paths.work_root of that config
    ORGANIC3D_CLUSTER_WORK_ROOT  a second work root holding runs trained / evaluated on another machine (e.g. an HPC
                                cluster: stage-A ablations, stage-B runs, NV-Reason-CT audit); default = the work root

Scripts call these at import (W = work_root()), so they need the config or the environment variables above.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV = os.path.join(HERE, '..', 'configs', 'env', 'local.yaml')


def env_paths(env=None):
    """paths section of the env config."""
    from omegaconf import OmegaConf
    env = env or os.environ.get('ORGANIC3D_ENV', DEFAULT_ENV)
    if not os.path.exists(env):
        raise FileNotFoundError(f'{env} not found: copy configs/env/TEMPLATE.yaml to configs/env/local.yaml (or set '
                                f'ORGANIC3D_ENV / ORGANIC3D_WORK_ROOT)')
    return OmegaConf.load(env).paths


def work_root():
    return os.environ.get('ORGANIC3D_WORK_ROOT') or str(env_paths().work_root)


def cluster_work_root():
    return os.environ.get('ORGANIC3D_CLUSTER_WORK_ROOT') or work_root()

