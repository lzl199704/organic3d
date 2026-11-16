#!/bin/bash
# Build the organic Python environment (conda, Python 3.12, torch 2.14 CUDA 12.6 build), then check it.
#   ENV_DIR=/path/to/envs/organic TORCH_CUDA=12 bash scripts/setup_env.sh           # torch CUDA 12.6 build (driver >= 525)
#   FRESH=1 ENV_DIR=... bash scripts/setup_env.sh                                     # delete and rebuild the env
# Needs internet and conda on PATH (on HPC systems load your conda module first). Optional: CACHE_DIR keeps pip / conda
# caches and temporary files out of a small home quota. Ends with scripts/check_env.py, which must print ENV OK.
set -eo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${ENV_DIR:?set ENV_DIR to the folder of the new conda env}
CACHE_DIR=${CACHE_DIR:-$(dirname "$ENV_DIR")}
export PIP_CACHE_DIR=$CACHE_DIR/pip_cache TMPDIR=$CACHE_DIR/tmp CONDA_PKGS_DIRS=$CACHE_DIR/conda_pkgs
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$CONDA_PKGS_DIRS" "$(dirname "$ENV_DIR")"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
if [ "${FRESH:-0}" = 1 ] && [ -d "$ENV_DIR" ]; then echo "== FRESH=1: removing $ENV_DIR"; rm -rf "$ENV_DIR"; fi
[ -x "$ENV_DIR/bin/python" ] || conda create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.12 pip   # conda-forge: no defaults-channel ToS prompt
conda activate "$ENV_DIR"
unset PYTHONPATH
echo "== python: $(which python) $(python --version)"
python -m pip install --upgrade pip wheel setuptools
TORCH_CUDA=${TORCH_CUDA:-12}
if [ -n "${TORCH_INDEX:-}" ]; then IDX="--index-url $TORCH_INDEX"
elif [ "$TORCH_CUDA" = 12 ]; then IDX="--index-url https://download.pytorch.org/whl/cu126"   # CUDA 12 build published for torch 2.14
else IDX=""; fi
python -m pip install torch==2.14.0 torchvision==0.29.0 $IDX
# everything else, without letting pip touch torch/torchvision again (it would pull the default CUDA 13 build)
grep -vE '^(torch|torchvision)==' "$CODE/requirements.txt" > "$TMPDIR/requirements_no_torch.txt"
python -m pip install -r "$TMPDIR/requirements_no_torch.txt"
python -c "import torch; print('== torch', torch.__version__, 'CUDA', torch.version.cuda, 'from', torch.__file__)"
python -m pip freeze > "$(dirname "$ENV_DIR")/organic3d_freeze.txt"
python "$CODE/scripts/check_env.py"
