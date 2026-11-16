# extaudit: the ORGANIC-3D benchmark harness

`extaudit` runs the alignment benchmark on any 3D CT image–text encoder or generative VLM. Every model gets the same
scans, the same anatomical interventions (organ erasure with fat / soft-tissue fill, left–right mirror, lesion erasure (earlier protocol)
with a mirrored control) and the same metric code. Our own models go through it too. The interventions edit the
1 mm RAS volume in Hounsfield units *before* the model's own preprocessing, so each model sees its native input
geometry. Each released model runs in its own Python environment with its authors' preprocessing and prompts. The
harness imports only numpy / torch / nibabel / pandas, so it runs inside any model environment. Adapters may import
more.

Metric definitions: `docs/metrics.md`. How to add a model:
`docs/adapter_guide.md`.

## Entry points

| Module | What it does | Output (under `--root`) |
|---|---|---|
| `python -m extaudit.core --task ladder` | Encoder ladder: Δ_f = score(positive prompt) − score(negative prompt) per finding on the clean scan, the mirrored scan and each organ erasure (fills `--fills`), plus the laterality sentence pairs | `out/<name>/<dataset>_shard{k}of{n}.npz` |
| `python -m extaudit.core --task grounding` | ReXGroundingCT clean finding scores for the lesion AUROC (`scripts/metrics/lesion_cls_metrics.py`; the mirrored-control erasure it also records is an earlier protocol, not LE); `--rex_set`, `--rex_exclude`, `--out_tag` select the scan set and output prefix (v2 runs: `--rex_set ctrate_rex_set_v2.parquet --out_tag grounding_v2`) | `out/<name>/grounding_shard{k}of{n}.parquet` (or `<out_tag>_shard...`) |
| `python -m extaudit.core --task aa3_det` | AbdomenAtlas 3.0 IID test (Table 1): liver, pancreas and kidney tumour detection with each model's own prompts and the free sentence pair (`extaudit/aa3_det.py`; set from `scripts/data/prepare_aa3.py`; metrics `scripts/metrics/lesion_det_metrics.py`, dataset value = mean of the three) | `out/<name>/aa3_det_shard{k}of{n}.parquet` |
| `python -m extaudit.core --task occlusion` | Evidence maps (occlusion sensitivity) for the qualitative figure (`extaudit/occlusion.py`, case table `$FIG4_CASES`) | `out/<name>/occlusion/<case>.parquet` |
| `LE_SET=<rex\|aa3> python -m extaudit.core --task lesion_erasure` | Lesion erasure (LE, Table 1): the lesion replaced by its organ's normal tissue vs the same edit of a same-shaped healthy region of the same organ (`extaudit/lesion_erasure.py`; sets from `scripts/data/prepare_heal.py`; metric `scripts/metrics/heal_metrics.py`) | `out/<name>/heal_<set>_shard{k}of{n}.parquet` |
| `SIDE_SET=<ctrate\|merlin> python -m extaudit.core --task side` | Laterality accuracy (Side, Table 1) on the original scan (`extaudit/side.py`; metric `scripts/metrics/side_metrics.py`) | `out/<name>/side_<set>_shard{k}of{n}.parquet` |
| `python -m extaudit.core --task chest` | CT-RATE chest tests: lung lobe erasure hit rate (LL) (each lobe erased in turn, 2,018 sentences naming one lobe) and density edits (emphysema, aortic and coronary calcium) (`extaudit/chest.py`; set from `scripts/data/prepare_chest_set.py`; metrics `scripts/metrics/chest_harness_metrics.py`) | `out/<name>/chest_shard{k}of{n}.npz` |
| `python -m extaudit.vlm_core` | Generative VLMs: P(Yes) (or the yes/no logit) per finding question on clean, mirrored and organ-erased scans; `--set_file rex_heldout_set.parquet --clean_only --out_tag rex_heldout` for the ReXGroundingCT classification (464 radiologist-annotated scans) | `out_vlm/<name>/<dataset>_shard{k}of{n}.npz` |
| `python -m extaudit.vlm_report` | Free-text reports on the full test set, greedy, each model's own report prompt (metrics `scripts/metrics/report_metrics_std.py`) | `<work_root>/eval_reports/<name>/<dataset>_reports_shard{k}of{n}.jsonl` |

All of them take `--model <adapter module>`, `--ckpt` (adapter-specific variant string), `--name` (output name),
`--dataset {ctrate, merlin, radchest, ...}`, `--root <work_root>/extaudit` and `--shard k/n`.

**Inputs.** `{root}/meta.json` holds the finding names, prompt phrases, the erased organ groups as
TotalSegmentator ids, the fill HU values and the noise SD. `{root}/<dataset>_set.parquet` has one row per test scan:
`index`, `volume`, `patient_id`, `img`, `seg` (absolute paths on your machine), `label_*` and `lat_pairs`. Encoders and VLMs
use the same sets. Both files are written by `scripts/data/export_audit_set.py`. Grounding needs
`scripts/data/prepare_rexgrounding.py` and LE needs `scripts/data/prepare_heal.py`.

**Metrics.** The shards hold per-scan scores only. Metrics are computed by `scripts/metrics/extaudit_metrics.py` (encoders),
`scripts/metrics/vlm_metrics.py` (VLMs), `scripts/metrics/lesion_cls_metrics.py`, `scripts/metrics/heal_metrics.py`, `scripts/metrics/side_metrics.py`,
`scripts/metrics/finding_reports.py` (finding-based reports) and `scripts/metrics/report_metrics_std.py` (free-text reports).

**Convenience wrappers.** `run_local.sh` runs a ladder over several GPUs and then computes its metrics,
`run_grounding.sh` runs ReXGroundingCT for a list of encoders. Set
`EXTAUDIT_ROOT=<work_root>/extaudit` first. Set `ORGANIC3D_PY` to the organic environment's python for the metric step.

## Where third-party files live

Adapters resolve repos and weights under `$ORGANIC3D_BASELINES` (default `<repo>/baselines`, see `extaudit/paths.py`),
in one folder per model as listed below. No third-party code or weights are included in this repository: clone and
download them yourself, under their own licences. `HF_HOME` defaults to `$ORGANIC3D_BASELINES/hf_cache` for the
adapters that set it.

## Adapters

Licences are copied from each adapter's docstring and checked against the local clones where we had one. **Check
every licence yourself before use. Several are non-commercial.**

### Encoders (`extaudit/models/`)

| Adapter | Model | Code | Weights | Licence (as found) | Local layout under `$ORGANIC3D_BASELINES` |
|---|---|---|---|---|---|
| `organic3d_enc` | ORGANIC-3D encoder (ours, stage A); `--ckpt colipri` = released COLIPRI | this repo | `zllhhhh/organic3d-encoder` (https://huggingface.co/zllhhhh/organic3d-encoder); COLIPRI: HF `microsoft/colipri` | ours: see top-level LICENSE; COLIPRI: MIT (per `configs/stageA.yaml` comment, verify) | none (organic env) |
| `ctclip` | CT-CLIP, CT-VocabFine (`--ckpt vocabfine`) | github.com/ibrahimethemhamamci/CT-CLIP | HF `ibrahimhamamci/CT-RATE` (`models/CT-CLIP_v2.pt`, `CT_VocabFine_v2.pt`) | CC BY-NC-SA 4.0 (README "License" section of the local clone; no LICENSE file) | `CT-CLIP/` (repo with `transformer_maskgit/`, `CT_CLIP/`, the two `.pt`), `CT-CLIP/ct_clip_venv` |
| `fvlm` | fVLM | github.com/alibaba-damo-academy/fvlm | README's Google Drive folder (`model.pth`) | no LICENSE file in repo (UNKNOWN) | `fvlm/repo`, `fvlm/weights/checkpoints/model.pth`, `fvlm/textenc` (CXR-BERT snapshot + config), `fvlm/venv` |
| `visd_boost` | ViSD-Boost | github.com/alibaba-damo-academy/ViSD-Boost | `checkpoint.pth`, `checkpoint_vqvae.pth` from the repo | no LICENSE file in repo (UNKNOWN) | `visd_boost/repo` (+ the two checkpoints); runs in `fvlm/venv` |
| `hlip` | HLIP ViT-B (CT-RATE) | github.com/Zch0414/hlip (tag v1.0) | v1.0 README's Google Drive link | MIT (LICENSE in local clone) | `hlip/hlip_repo`, `hlip/weights/hlip_ctrate_vitb.pt`, `hlip/venv` |
| `itemizedclip` | ItemizedCLIP (CT-RATE) | github.com/MLNeurosurg/ItemizedCLIP | README's Google Drive link | MIT (LICENSE in local clone) | `hlip/itemizedclip_repo`, `hlip/weights/itemizedclip_ctrate.pt`; runs in `hlip/venv` |
| `merlin` | Merlin | github.com/StanfordMIMI/Merlin (pip `merlin-vlm`) | HF `stanfordmimi/Merlin` | MIT (docstring) | `merlin/venv`; optional `MERLIN_ROOT` (see adapter docstring) |
| `spectre` | SPECTRE-Large | github.com/cclaess/SPECTRE @c05af47 | HF `cclaess/SPECTRE-Large`, `cclaess/SPECTRE` | code MIT (LICENSE), weights CC BY-NC-SA 4.0 (LICENSE_MODELS) | `spectre/components` (released backbone, combiner, SigLIP heads, LoRA), `spectre/venv` |
| `jolia` | Jolia (`--ckpt concat`, `cls`, `long`, `long_cls`) | HF `raidium/Jolia` (gated, remote code) | HF `raidium/Jolia`; text tower HF `Qwen/Qwen3-Embedding-8B` | Research-Only RAIL-M (LICENSE in HF repo) | `jolia/hf` (clone of the HF repo), `jolia/text_cache.pt` (optional), `jolia/venv` |
| `radar` | RADAR+ (`--ckpt plus` = Merlin-only, `plus_ft` = published) | github.com/alibaba-damo-academy/damo-radar | huggingface.co/radar-generalist | code Apache-2.0 (LICENSE); README and weights CC BY-NC-SA 4.0 | `radar/damo-radar`, `radar/work` (empty), `radar/ckpt` (weights + `bert-base-uncased/` + the repo's `ckpt/infer_text_embedding_*.pt`), `radar/venv` |

### Generative VLMs (`extaudit/vlm_models/`, plus two audits in `organic/`)

| Adapter | Model | Code | Weights | Licence (as found) | Local layout |
|---|---|---|---|---|---|
| `organic3d_vlm` | ORGANIC-3D VLM (ours, stage B) | this repo | `zllhhhh/organic3d-vlm` (https://huggingface.co/zllhhhh/organic3d-vlm); LLM HF `Qwen/Qwen3.5-4B` | ours: top-level LICENSE; Qwen3.5: check the model card | organic env |
| `m3d_lamed` | M3D-LaMed-Phi-3-4B | github.com/BAAI-DCAI/M3D | HF `GoodBaiBai88/M3D-LaMed-Phi-3-4B` | code MIT (LICENSE in clone), weights Apache-2.0 (model card) | `m3d_lamed/weights` (HF snapshot), `m3d_lamed/venv` |
| `radfm` | RadFM | github.com/chaoyi-wu/RadFM | HF `chaoyi-wu/RadFM` (`pytorch_model.zip`) | MIT (LICENSE in clone); weights licence UNKNOWN | `radfm/repo`, `radfm/weights/radfm_bf16.safetensors` (bf16 conversion of the released checkpoint, see open items), `radfm/venv` |
| `omnict` | OmniCT-7B | github.com/alibaba-damo-academy/OmniCT @d9ddc8b | HF `Alibaba-DAMO-Academy/OmniCT-7B`, `google/siglip-so400m-patch14-384` | weights Apache-2.0 (model card); code: no LICENSE file in repo (UNKNOWN) | `omnict/repo`, `omnict/weights/{OmniCT-7B, siglip-so400m-patch14-384}`, `omnict/venv` |
| `hulu_med` | Hulu-Med-7B | HF remote code (`ZJU-AI4H/Hulu-Med-7B` @258594714a) | same | Apache-2.0 (docstring; LICENSE in local clone of the code repo) | `hulu_med/venv` |
| `ctchat` | CT-CHAT (yes/no answers and free-text reports) | CT-CHAT code (llava) from the CT-RATE authors | HF `ibrahimhamamci/CT-RATE` (`models/CT-CHAT`), CT-CLIP v2 | CC BY-NC-SA 4.0 (README "License" section; no LICENSE file) | `ctchat/code`, `ctchat/weights`, `ctchat/ctchat_score.py` (not distributed, see open items), `ctchat/venv`; needs `CT-CLIP/` |
| `medgemma` | MedGemma 1.5 4B (yes/no answers and free-text reports) | `organic/audit_medgemma.py` | HF `google/medgemma-1.5-4b-it` | Health AI Developer Foundations terms (UNKNOWN here, check the model card) | organic env |
| `organic/audit_medgemma.py` | MedGemma 1.5 4B model wrapper used by the `medgemma` adapter | this repo | as above | as above | organic env |
| `nv_reason` | NV-Reason-CT through the shared harness (yes/no answers on clean, mirrored and erased scans; reports; the paper's `NV-Reason-CT_h` rows) | wraps `organic/audit_nvreason.py` | HF `nvidia/NV-Reason-CT` | UNKNOWN (check the model card) | organic env |
| `organic/audit_nvreason.py` | NV-Reason-CT model wrapper (preprocessing, prompts, generation) used by the `nv_reason` adapter | this repo | HF `nvidia/NV-Reason-CT` | UNKNOWN (not stated in the docstring; check the model card) | organic env |

## Per-baseline environments

Each model was run in its own environment, built from its upstream instructions. The versions below are what we
used. Package names follow pip.

| Environment | Python | Key packages |
|---|---|---|
| organic (ours, COLIPRI, MedGemma, NV-Reason-CT) | 3.10 (3.12 also used) | torch 2.14, transformers 5.17, peft 0.21.1, monai 1.5.1, timm 1.0.22, colipri 0.1.3, flash-linear-attention 0.5.2 (see `requirements.txt`) |
| CT-CLIP | 3.10 | torch 2.5.0, transformers 4.30.1, monai 1.3.2, CT-CLIP repo (`transformer_maskgit`, `CT_CLIP`) |
| fVLM / ViSD-Boost | 3.11 | torch 2.3.1+cu121, transformers 4.25.0, monai 1.3.2 |
| HLIP / ItemizedCLIP | 3.11 | torch 2.5.1+cu121, transformers 4.49.0, open_clip_torch 2.32.0, timm 1.0.30 |
| Merlin | 3.10 | torch 2.3.1+cu121, transformers 4.44.2, monai 1.3.2, peft 0.11.1, `merlin-vlm` (the env holds 0.0.7; the docstring says 2.x, so check) |
| SPECTRE | 3.11 | torch 2.8.0+cu128, transformers 5.3.0, timm 1.0.30, SPECTRE repo @c05af47 (+ loralib) |
| Jolia | 3.11 | torch 2.8.0+cu128, transformers 5.3.0, timm 1.0.30, the HF repo's remote code |
| RADAR+ | 3.10 | torch 2.3.1+cu121, transformers 4.25.0, monai 1.3.2, dynamic_network_architectures, damo-radar `RADAR_train` |
| Hulu-Med | 3.10 | torch 2.4.0+cu121, transformers 4.51.2, flash-attn 2.7.3, timm 1.0.3 |
| M3D-LaMed | 3.10 | torch 2.4.1, transformers 4.42.4, monai 1.3.2 |
| RadFM | 3.10 | torch 2.5.1+cu124, transformers 4.44.2 |
| OmniCT | 3.11 | torch 2.7.1+cu128, transformers 4.57.3, peft 0.18.1, monai 1.3.0, OmniCT repo @d9ddc8b (SDPA attention, no flash-attn) |
| CT-CHAT | 3.11 | transformers 4.44.0, peft 0.12.0, timm 0.9.16, the repo's llava code, CT-CLIP `transformer_maskgit` |

Every environment additionally needs numpy, nibabel, pandas, pyarrow and zstandard for the harness.

## Open items (not distributed)

- **CT-CHAT helper.** The `ctchat` adapter's import of `preprocess`, `BASE` and `MODEL` come from a script that is not part of
  this repository. Place it at `$ORGANIC3D_BASELINES/ctchat/ctchat_score.py`.
- **Jolia text cache.** Template prompts are encoded lazily, so no cache is needed for them. The `long` / `long_cls`
  modes need prototype reports (50 positive and 50 negative training reports per finding) in
  `$ORGANIC3D_BASELINES/jolia/text_cache.pt`. The cache builder is not distributed.
- **RadFM bf16 weights.** `radfm_bf16.safetensors` is a one-off bf16 conversion of the released checkpoint. It drops
  the Med-KEBERT keyword-loss weights and rotary buffers. The conversion helper is not distributed.
