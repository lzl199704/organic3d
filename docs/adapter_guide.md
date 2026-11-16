# Adapter guide: plugging a new 3D CT encoder or VLM into the benchmark

The harness (`extaudit/`) owns the scans, the interventions and the metric code. An adapter owns only what is specific
to one model: loading it, its preprocessing, its prompts and its score. Adapters run inside the model's own Python
environment; the harness itself imports only numpy, torch, nibabel and pandas (plus zstandard for `.bin` volumes).

Worked examples: `extaudit/models/ctclip.py` (minimal encoder), `extaudit/models/organic3d_enc.py` (mask-guided crop via
`set_scan`, custom `similarity`), `extaudit/models/merlin.py` (`set_volume`), `extaudit/vlm_models/m3d_lamed.py`
(minimal VLM), `extaudit/vlm_models/omnict.py` (`yes_no_logit`, batched questions, free-text report).

## What the adapter receives

`hu`: a `torch.float32` CUDA tensor of shape (X, Y, Z) in **HU, 1 mm isotropic, RAS array order** (nibabel
`aff2axcodes` = R, A, S): axis 0 runs from the patient's left to right (the left lung sits at small x), axis 1 from
posterior to anterior, axis 2 from inferior to superior. The volume may already be edited (organ erased, mirrored,
lesion healed): the adapter must not care.

`seg` (optional, via `set_scan(seg)`): TotalSegmentator v2 "total" label ids on the same grid (lungs 10–14, heart 51,
aorta 52, liver 5, kidneys 2/3, ...). Under the mirror condition the harness passes the mirrored mask. Use it only if
the model's official pipeline needs masks (e.g. organ-gated models) or to position a crop.

## Encoder API (`extaudit/models/<name>.py`, run with `python -m extaudit.core --model <name>`)

```python
class Adapter:
    def __init__(self, ckpt, device): ...           # ckpt = the --ckpt string (a variant name or a path), may be None
    @staticmethod
    def prompts(f, meta): ...                       # -> (positive, negative) sentence for finding id f
                                                    #    meta['phrase'][f], meta['label_name'][f] help build them
    def encode_text(self, sents): ...               # list[str] -> (S, D) tensor
    def encode_image(self, hu): ...                 # -> (state, feat): state = whatever similarity() needs,
                                                    #    feat = 1-D pooled image feature (linear probe, supplementary)
    # optional
    def similarity(self, state, t): ...             # (S, D) text -> (S,) scores; default: t @ state
    def set_scan(self, seg): ...                    # mask of the current scan (called before its conditions)
    def set_volume(self, volume): ...               # volume id of the current scan (e.g. to restore a field of view)
    def set_dataset(self, name): ...                # 'ctrate' | 'merlin' | 'radchest' (chest vs abdominal crop)
    max_len = 96                                    # if present, the retrieval task raises it to 512 for whole reports
```

The harness computes $\Delta_f$ = `similarity(state, encode_text([pos]))` − `similarity(state, encode_text([neg]))`. If
the model has only one prompt per finding, return a null negative whose score is exactly 0 (see
`extaudit/models/itemizedclip.py`). Free sentences (laterality pairs, lesion sentences, reports) go through the same
`encode_text` / `similarity`.

## VLM API (`extaudit/vlm_models/<name>.py`, run with `python -m extaudit.vlm_core --model <name>`)

```python
class Adapter:
    def __init__(self, ckpt, device): ...
    def encode(self, hu): ...                       # -> (state, feat); state = e.g. visual tokens, feat = 1-D feature
    def p_yes(self, state, questions): ...          # -> (n,) P(Yes) = sigmoid(logit Yes - logit No), first answer token
    # optional
    def question(self, phrase): ...                 # the model's own yes/no question (default: vlm_core.default_question)
    def yes_no_logit(self, state, questions): ...   # the logit difference itself (preferred: no ties when P saturates)
    def set_scan(self, seg): ...
    def set_dataset(self, name): ...
    # free-text reports (python -m extaudit.vlm_report)
    def report_prompt(self, dataset): ...           # the model's own report request
    def report(self, hu, dataset, max_new_tokens): ...   # greedy free-text report
```

## Rules

1. **Reproduce the official preprocessing** from the model's repository: spacing, crop / pad size, HU window,
   scaling, channel layout and **orientation**. Models trained on CT-RATE NIfTI files usually expect the CT-RATE file
   order (our RAS array flipped on axes 0 and 1; see `ctclip.py`). If the orientation is not documented, test the
   variants on ~150 CT-RATE scans and keep the one that best matches the published numbers; write what you found in
   the module docstring.
2. **Prompts**: use the model's own published zero-shot prompt or question format and cite where it comes from. If
   none exists, use the harness default (`There is X.` / `No X.` for encoders; `Is there X in this CT? Answer Yes or
   No.` for VLMs).
3. **Environment**: one environment per model, built from its upstream instructions; it needs numpy, torch, nibabel,
   pandas, pyarrow and zstandard in addition to the model's own packages. Put repositories and weights under
   `$ORGANIC3D_BASELINES/<model>/` and resolve them with `extaudit.paths.baseline(...)`; never hard-code machine paths.
   Do not vendor third-party code into this repository.
4. **Docstring**: state the model and paper, the code and weight sources with their licences, the exact preprocessing
   and its source, the prompt format, the score rule, and any deviation from the official pipeline.

## Verify, then run

```bash
export EXTAUDIT_ROOT=<work_root>/extaudit            # meta.json and *_set.parquet from scripts/data/export_audit_set.py
# encoder smoke test: 150 CT-RATE scans, fat fill only
CUDA_VISIBLE_DEVICES=0 <model env>/bin/python -m extaudit.core --model <name> --name smoke_<name> \
    --dataset ctrate --root $EXTAUDIT_ROOT --limit 150 --fills fat --workers 4
# VLM smoke test: 60 scans of the 1,000-scan audit sample
CUDA_VISIBLE_DEVICES=0 <model env>/bin/python -m extaudit.vlm_core --model <name> --name smoke_<name> \
    --dataset ctrate --root $EXTAUDIT_ROOT --limit 60
```

Sanity checks before a full run: the per-finding zero-shot AUROC of `delta__clean` (encoders) or `P__clean` (VLMs) in
the shard `.npz` against the `label_<finding>` columns of the set parquet (row `index`) should be close to the
published number for the model; for VLMs, check the spread of P(Yes) (a constant answer means a broken prompt). Then
run the full sets (`--shard k/n` splits the work across GPUs) and the metrics:

```bash
bash extaudit/run_local.sh <name> <ckpt|-> <Display-Name> ctrate "0 1" 2 <model env>/bin/python   # ladder + metrics
python scripts/metrics/extaudit_metrics.py --env configs/env/local.yaml --name <Display-Name> --dataset ctrate
python scripts/metrics/vlm_metrics.py --env configs/env/local.yaml --name <Display-Name> --datasets ctrate,merlin
```

Tasks beyond the ladder need no adapter changes: `--task grounding` (ReXGroundingCT), `--task lesion_erasure` and `--task aa3_det` (the
abdominal sets put the adapter in abdominal mode via `set_dataset('merlin')`), `--task side`.
