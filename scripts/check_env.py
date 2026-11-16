"""Check the Python environment without a GPU: versions against requirements.txt, every import our code uses,
the linear-attention kernels Qwen3.5 needs, a small LoRA and MONAI test, and the GPU driver the torch build needs.
Prints ENV OK or the failed checks.       python scripts/check_env.py
"""
import importlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, fn):
    try:
        out = fn()
        print(f'[ok]   {name}' + (f': {out}' if out else ''), flush=True)
    except Exception as e:
        fails.append(name)
        print(f'[FAIL] {name}: {type(e).__name__}: {e}', flush=True)


def versions():
    from importlib.metadata import version, PackageNotFoundError
    bad = []
    for line in open(os.path.join(HERE, '..', 'requirements.txt')):
        m = re.match(r'^([A-Za-z0-9_.\-]+)==([^\s#]+)', line.strip())
        if not m:
            continue
        pkg, want = m.groups()
        try:
            have = version(pkg)
        except PackageNotFoundError:
            bad.append(f'{pkg} missing'); continue
        if have.split('+')[0] != want:
            bad.append(f'{pkg} {have} (want {want})')
    assert not bad, '; '.join(bad)
    return f'python {sys.version.split()[0]}, all pinned packages match'


def imports():
    mods = ['torch', 'torchvision', 'triton', 'numpy', 'pandas', 'pyarrow', 'scipy', 'sklearn', 'nibabel', 'omegaconf',
            'einops', 'timm', 'safetensors', 'transformers', 'tokenizers', 'huggingface_hub', 'peft', 'accelerate', 'monai',
            'colipri', 'dynamic_network_architectures', 'torchio']
    for m in mods:
        importlib.import_module(m)
    sys.path.insert(0, os.path.join(HERE, '..'))
    for m in ['organic.vocab', 'organic.config', 'organic.data', 'organic.augment', 'organic.model', 'organic.losses',
              'organic.train_stageA', 'organic.evaluate_stageA', 'organic.vlm', 'organic.train_stageB',
              'organic.evaluate_stageB', 'organic.ladder', 'organic.audit_nvreason']:
        importlib.import_module(m)
    return f'{len(mods)} packages and 13 project modules import'


def kernels():
    import fla  # noqa: F401  (fails on Python 3.10 without scripts/patch_triton_py310.py)
    import torch
    if not torch.cuda.is_available():
        # Triton kernels need a GPU; on a CPU-only login node fla falls back to CPU and transformers reports them
        # unavailable. The GPU validation job (lsf_validate_gpu.bsub, step 1) runs this check again on an H100.
        return 'fla imports; no GPU here, kernel check deferred to the GPU validation job'
    from transformers.utils import import_utils as iu
    assert iu.is_flash_linear_attention_available(), 'transformers does not see flash-linear-attention'
    from transformers.models.qwen3_5 import modeling_qwen3_5  # noqa: F401
    return 'flash-linear-attention loads; Qwen3.5 linear attention will use it'


def torch_cpu():
    import torch
    x = torch.randn(64, 64)
    assert torch.allclose(x @ torch.eye(64), x)
    cuda = torch.version.cuda
    need = {'13.0': 580, '12.9': 575, '12.8': 570, '12.6': 560}.get(cuda, 525)
    return f'torch {torch.__version__}, built for CUDA {cuda}: needs NVIDIA driver >= {need} on the GPU nodes'


def lora():
    import torch
    from peft import LoraConfig, get_peft_model
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.q_proj = torch.nn.Linear(8, 8)
        def forward(self, x):
            return self.q_proj(x)
    m = get_peft_model(Tiny(), LoraConfig(r=2, target_modules=['q_proj']))
    assert m(torch.randn(2, 8)).shape == (2, 8)
    return f'peft LoRA ok ({sum(p.numel() for p in m.parameters() if p.requires_grad)} trainable)'


def monai_tf():
    from monai.transforms import Compose, EnsureTyped, Orientationd, Spacingd  # noqa: F401
    return 'MONAI transforms import'


def isolation():
    import torch, triton, transformers
    assert not os.environ.get('PYTHONPATH'), f"PYTHONPATH is set ({os.environ['PYTHONPATH'][:80]}...): run 'unset PYTHONPATH'"
    import numpy
    pre = os.path.realpath(sys.prefix)
    bad = [m.__name__ for m in (torch, triton, transformers, numpy) if not os.path.realpath(m.__file__).startswith(pre)]
    assert not bad, f'{bad} load from outside this environment ({sys.prefix})'
    code = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    base = os.path.realpath(sys.base_prefix)
    def foreign(p):
        r = os.path.realpath(p)
        if r.startswith(pre) or r.startswith(code):
            return False
        if r.startswith(base) and 'site-packages' not in r and 'dist-packages' not in r:
            return False                     # the base interpreter's standard library (normal for a venv)
        return True
    leak = [p for p in sys.path if p and os.path.isdir(p) and foreign(p)]
    assert not leak, f'search path contains folders outside the env: {leak[:3]}'
    return f'torch, triton, transformers, numpy and the whole search path are inside {sys.prefix}'


for name, fn in [('isolation', isolation), ('versions', versions), ('imports', imports), ('linear-attention kernels', kernels),
                 ('torch (CPU) and CUDA build', torch_cpu), ('LoRA', lora), ('MONAI', monai_tf)]:
    check(name, fn)
print('ENV OK' if not fails else f'ENV FAILED: {fails}')
sys.exit(1 if fails else 0)
