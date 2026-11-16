#!/bin/bash
# Run one released encoder through the extaudit ladder on local GPUs, then its metrics.
#   EXTAUDIT_ROOT=<work_root>/extaudit bash extaudit/run_local.sh <adapter> <ckpt|-> <name> <dataset> "<gpus>" <slots per gpu> <python>
# e.g. bash extaudit/run_local.sh ctclip ctclip CT-CLIP ctrate "0 1" 2 $ORGANIC3D_BASELINES/CT-CLIP/ct_clip_venv/bin/python
# FILLS=fat (env) runs only the fat-fill erasures (default fat,soft). ORGANIC3D_PY = python of the organic env (metrics),
# ORGANIC3D_ENV = data-root config (default configs/env/local.yaml).
cd "$(dirname "$0")/.."
AD=$1; CK=$2; NAME=$3; DS=$4; GPUS=($5); SLOTS=$6; PY=$7
ROOT=${EXTAUDIT_ROOT:?set EXTAUDIT_ROOT=<work_root>/extaudit}
EP=${ORGANIC3D_PY:-python}; ENVF=${ORGANIC3D_ENV:-configs/env/local.yaml}
N=$(( ${#GPUS[@]} * SLOTS )); mkdir -p logs
FILLS=${FILLS:-fat,soft}
ARGS="--model $AD --name $NAME --dataset $DS --root $ROOT --workers 4 --fills $FILLS"; [ "$CK" != - ] && ARGS="$ARGS --ckpt $CK"
i=0; for g in "${GPUS[@]}"; do for s in $(seq 1 $SLOTS); do
  [ -f $ROOT/out/$NAME/${DS}_shard${i}of$N.npz ] || CUDA_VISIBLE_DEVICES=$g $PY -W ignore -m extaudit.core $ARGS --shard $i/$N > logs/extaudit_${NAME}_${DS}_$i.log 2>&1 &
  i=$((i+1)); done; done; wait
grep -h "^DONE\|ERROR" logs/extaudit_${NAME}_${DS}_*.log
$EP scripts/extaudit_metrics.py --env $ENVF --name $NAME --dataset $DS --fills $FILLS > logs/extaudit_${NAME}_${DS}_metrics.log 2>&1 && grep SUMMARY logs/extaudit_${NAME}_${DS}_metrics.log
