"""Where third-party baselines live on this machine (repos, weights, per-model Python environments).

Every adapter in extaudit/models and extaudit/vlm_models resolves its local files under one root,

    ORGANIC3D_BASELINES   (environment variable; default: <repo>/baselines)

with one sub-folder per model, laid out as described in extaudit/README.md (e.g. $ORGANIC3D_BASELINES/hlip/hlip_repo,
$ORGANIC3D_BASELINES/hlip/weights/hlip_ctrate_vitb.pt). Hugging Face downloads go to $HF_HOME when it is set, else to
$ORGANIC3D_BASELINES/hf_cache. Nothing here is imported by the metric code, so any model environment can use it.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = os.path.abspath(os.path.expanduser(os.environ.get('ORGANIC3D_BASELINES', os.path.join(REPO, 'baselines'))))


def baseline(*parts):
    """Path below the baselines root, e.g. baseline('hlip', 'weights', 'hlip_ctrate_vitb.pt')."""
    return os.path.join(BASELINES, *parts)


def default_hf_home():
    """HF cache shared by the baseline environments (only used when HF_HOME is not already set)."""
    return baseline('hf_cache')
