# Scripts

Run every script from the repository root (`python scripts/<folder>/<script>.py`). Locations come from the env file
(`configs/env/local.yaml`, or `$ORGANIC3D_ENV` / `$ORGANIC3D_WORK_ROOT`, see `organic/workroot.py`). Per-scan scores go to
`{work_root}/extaudit/out*/<name>/`, metrics to `{work_root}/eval/` and `{work_root}/evalB/<name>/`.

Training and model evaluation are modules of the `organic` package (`python -m organic.encoder.train`, see
`organic/README.md`); the scripts here prepare data, launch runs, run the benchmark and compute the metrics.

| Folder | What it holds | Start with |
|---|---|---|
| `setup/` | environment build and checks, CPU preflight, GPU smoke test | `setup_env.sh`, `check_env.py` |
| `data/` | 1 mm conversion, splits, manifests, report sentences, training cache, external test sets, benchmark sets | `docs/data_preparation.md` |
| `train/` | multi-GPU and multi-node training launchers | `launch_local.sh` |
| `evaluation/` | one command per model for every test of the paper (report metrics in their own environment, see `requirements.txt`); score export for the Hugging Face benchmark | `evaluate.py` |
| `metrics/` | the metric behind each table, on the full test sets | `docs/metrics.md` |
| `analysis/` | anatomy-token attention maps (supplement) | `slot_attention_maps.py` |
| `inference/` | ORGANIC-3D encoder and VLM on one CT (the same file ships with the weights) | `organic3d.py` |
