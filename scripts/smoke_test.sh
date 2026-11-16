#!/bin/bash
# End-to-end check on a new machine: data paths, stage A (1 GPU, multi-GPU, resume, evaluation), stage B (train + evaluate).
#   ENV=configs/env/local.yaml GPUS=0,1,2,3 bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${ENV:-configs/env/local.yaml}
GPUS=${GPUS:-0,1,2,3}
PY=${PY:-$(dirname "$(command -v python)")}
FIRST=$(echo "$GPUS" | cut -d, -f1)
echo "== 1. paths and manifests"
"$PY/python" - "$ENV" <<'EOF'
import sys, os, pandas as pd
sys.path.insert(0, '.')
from organic.config import load_config, abspath
cfg = load_config('configs/stageA.yaml', env=sys.argv[1])
for d in ('merlin', 'ctrate'):
    m = pd.read_parquet(os.path.join(cfg.paths.work_root, 'manifests', f'{d}.parquet'))
    r = m[m.split == 'train'].iloc[0]
    # raw files, or the cache file when the dataset is read only through its cache (built elsewhere and copied)
    raw = all(os.path.exists(abspath(cfg, d, r[k])) for k in ('img_rel', 'seg_rel'))
    cache = os.path.join(cfg.paths.work_root, 'cache', d, f'{r.volume}.npz')
    assert raw or os.path.exists(cache), f'{d}: neither {abspath(cfg, d, r.img_rel)} nor {cache}'
    print(d, m.split.value_counts().to_dict(), 'files ok' if raw else 'cache only')
EOF
common="data.limit=64 data.num_workers=2 train.log_every=1 eval.n_val=16 eval.min_pos=1 train.resume=true"
echo "== 2. single GPU, full model, 4 steps"
CUDA_VISIBLE_DEVICES=$FIRST "$PY/python" -m organic.train_stageA --env "$ENV" --ablation configs/encoder/enc4_organic3d.yaml \
    run_name=smoke_1gpu train.max_steps=4 train.eval_every=4 train.save_every=4 $common
echo "== 3. all GPUs, 6 steps, then resume to 8"
GPUS=$GPUS ENV=$ENV PY=$PY bash scripts/launch_local.sh configs/encoder/enc4_organic3d.yaml run_name=smoke_multi train.max_steps=6 train.eval_every=6 train.save_every=3 $common
GPUS=$GPUS ENV=$ENV PY=$PY bash scripts/launch_local.sh configs/encoder/enc4_organic3d.yaml run_name=smoke_multi train.max_steps=8 train.eval_every=8 train.save_every=4 $common
echo "== 4. stage-A ladder evaluation (released COLIPRI and the smoke checkpoint)"
CUDA_VISIBLE_DEVICES=$FIRST "$PY/python" -m organic.evaluate_stageA --env "$ENV" --ckpt colipri --name smoke_colipri --datasets merlin,ctrate --limit 24 --min_pos 1
W=$("$PY/python" -c "import sys; sys.path.insert(0,'.'); from organic.config import load_config; print(load_config('configs/stageA.yaml', env='$ENV').paths.work_root)")
echo "== 5. stage B: 4 steps with every term on (dependence margin, DPO, read-out), then evaluation with report generation"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$FIRST "$PY/python" -m organic.train_stageB --env "$ENV" --ablation configs/vlm/vlm4_organic3d.yaml \
    run_name=smokeB vlm.max_steps=4 vlm.eval_every=4 vlm.save_every=4 vlm.n_val=8 vlm.log_every=1 vlm.resume=false \
    lossB.dpo=0.5 data.limit=64 data.num_workers=2 vlm.encoder_ckpt=$W/runs/smoke_multi/last.pt vlm.freeze_encoder=all
CUDA_VISIBLE_DEVICES=$FIRST "$PY/python" -m organic.evaluate_stageB --env "$ENV" --run $W/runsB/smokeB --datasets merlin,ctrate --limit 16 --gen 2
echo "SMOKE TEST PASSED"
