#!/bin/bash
# ReXGroundingCT finding scores (d_clean, for the lesion AUROC) for one encoder on local GPUs: shards in parallel,
# resumable per shard. Pass --rex_set ctrate_rex_set_v2.parquet --out_tag grounding_v2 through EXTRA for the full v2 set.
#   EXTAUDIT_ROOT=<work_root>/extaudit bash extaudit/run_grounding.sh <adapter> <ckpt|-> <name> "<gpus>" <slots per gpu> <python>
# Metrics afterwards: scripts/metrics/lesion_cls_metrics.py.
cd "$(dirname "$0")/.."
AD=$1; CK=$2; NAME=$3; GPUS=($4); SLOTS=$5; PY=$6
ROOT=${EXTAUDIT_ROOT:?set EXTAUDIT_ROOT=<work_root>/extaudit}
N=$(( ${#GPUS[@]} * SLOTS )); mkdir -p logs
ARGS="--model $AD --name $NAME --task grounding --root $ROOT --workers 4 ${EXTRA:-}"; [ "$CK" != - ] && ARGS="$ARGS --ckpt $CK"
i=0; for g in "${GPUS[@]}"; do for s in $(seq 1 $SLOTS); do
  [ -f $ROOT/out/$NAME/grounding_shard${i}of$N.parquet ] || CUDA_VISIBLE_DEVICES=$g $PY -W ignore -m extaudit.core $ARGS --shard $i/$N > logs/grounding_${NAME}_$i.log 2>&1 &
  i=$((i+1)); done; done; wait
grep -h "^DONE\|ERROR" logs/grounding_${NAME}_*.log
