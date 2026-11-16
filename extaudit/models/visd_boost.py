"""ViSD-Boost (Cao et al., ICCV 2025; github.com/alibaba-damo-academy/ViSD-Boost, checkpoint.pth + checkpoint_vqvae.pth
from the repo, no LICENSE file). Same organ-level pipeline, prompts and score rule as fVLM: see extaudit/models/fvlm.py
(ViSD differences: roi 112 x 256 x 352, window padded to a multiple of 16 x 32 x 32, 'Lung nodule.' / 'Lung opacity.'
prompts, ResNet-18 multi-scale tokens + anatomy-normality VQ-VAE branch, absent organ -> NaN as in its calc_metrics).
Run in $ORGANIC3D_BASELINES/fvlm/venv.
"""
from extaudit.models.fvlm import Adapter as _Adapter


class Adapter(_Adapter):
    def __init__(self, ckpt, device):
        super().__init__(ckpt or 'visd_boost', device)
