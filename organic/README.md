# organic

The ORGANIC-3D package. Training and evaluation run as modules from the repository root, e.g.
`torchrun --nproc_per_node 4 -m organic.encoder.train --env configs/env/local.yaml --ablation configs/encoder/enc4_organic3d.yaml`.

| Path | What it holds | Run as |
|---|---|---|
| `vocab.py` | organ groups (TotalSegmentator v2 ids), the finding vocabulary, target and control organs per finding | |
| `config.py`, `workroot.py` | config loading (base + env + run config + overrides); data and output locations | |
| `metrics.py` | matched-pair AUROC, clinical F1, report label extractors | |
| `utils.py` | distributed helpers (torchrun), AUROC | |
| `data/` | dataset and sampler (`dataset.py`), interventions: organ erasure, mirror, text edits (`interventions.py`), volume I/O (`volio.py`), CT-RATE v2 test list (`ctrate_v2.py`), Merlin report label rules (`label_rules.py`) | |
| `encoder/` | ORGANIC-3D encoder (`model.py`), fixed organ routing (`routing.py`), losses (`losses.py`) | `organic.encoder.train`, `organic.encoder.evaluate`, `organic.encoder.evaluate_chest` |
| `vlm/` | ORGANIC-3D VLM: encoder tokens, projector, Qwen3.5-4B with LoRA (`model.py`) | `organic.vlm.train`, `organic.vlm.evaluate` |
| `baselines/` | NV-Reason-CT and MedGemma wrappers (preprocessing, prompts, scoring) used by `extaudit/vlm_models/` | `organic.baselines.nv_reason_ct`, `organic.baselines.medgemma` |
