#!/bin/bash
# Single-node stage-A launch.
#   GPUS=0,1,2,3 ENV=configs/env/local.yaml bash scripts/launch_local.sh configs/encoder/enc4_organic3d.yaml [key=value ...]
set -euo pipefail
cd "$(dirname "$0")/.."
ABL=${1:?ablation yaml}; shift
GPUS=${GPUS:-0,1,2,3}
ENV=${ENV:-configs/env/local.yaml}
PY=${PY:-$(dirname "$(command -v python)")}     # bin folder of the organic env
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=4 NCCL_ASYNC_ERROR_HANDLING=1
exec "$PY/torchrun" --standalone --nproc_per_node "$NPROC" -m organic.train_stageA --env "$ENV" --ablation "$ABL" "$@"
