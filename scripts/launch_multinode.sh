#!/bin/bash
# Multi-node stage-A launch with torchrun (run the same command on every node).
#   NNODES=3 NODE_RANK=<0|1|2> MASTER_ADDR=<node0 host> ENV=configs/env/local.yaml \
#       bash scripts/launch_multinode.sh configs/encoder/enc4_organic3d.yaml train.batch_size=4
# 3 nodes x 8 H100 = 24 GPUs. For three parallel 8-GPU ablations use launch_local.sh on each node instead.
set -euo pipefail
cd "$(dirname "$0")/.."
ABL=${1:?ablation yaml}; shift
: "${NNODES:?}" "${NODE_RANK:?}" "${MASTER_ADDR:?}"
MASTER_PORT=${MASTER_PORT:-29511}
NPROC=${NPROC:-8}
ENV=${ENV:-configs/env/local.yaml}
PY=${PY:-$(dirname "$(command -v python)")}
export OMP_NUM_THREADS=4 NCCL_ASYNC_ERROR_HANDLING=1
exec "$PY/torchrun" --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC" \
    --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" --rdzv_id "${RUN_ID:-organic}" \
    -m organic.train_stageA --env "$ENV" --ablation "$ABL" "$@"
