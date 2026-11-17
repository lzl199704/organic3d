# Data preparation

No data are distributed with this repository. Download each dataset from its source under its own licence and terms
(see the README). All paths come from your env file:

```bash
cp configs/env/TEMPLATE.yaml configs/env/local.yaml   # then edit the roots
```

Everything is converted to one convention: **1 mm isotropic NIfTI, RAS array order, HU, with a TotalSegmentator v2
"total" multilabel mask on the same grid**, below each dataset root as `1mm/img/<id>.nii.gz` and
`1mm/ts_seg/<id>.nii.gz`. Derived tables (manifests, parsed sentences, caches, harness sets) go to `paths.work_root`.

## 1. Volumes and TotalSegmentator masks

```bash
# resample to 1 mm RAS (CPU)
python scripts/data/prepare_1mm.py --stage resample --src_dir <CT-RATE>/dataset/train --dst_root <ctrate_root> --workers 16
python scripts/data/prepare_1mm.py --stage resample --src_dir <CT-RATE>/dataset/valid --dst_root <ctrate_root> --workers 16
python scripts/data/prepare_1mm.py --stage resample --src_dir <Merlin images>          --dst_root <merlin_root> --workers 16
# TotalSegmentator v2 masks (in a TotalSegmentator environment; several workers can share the list)
<TotalSegmentator env>/bin/python scripts/data/prepare_1mm.py --stage ts --dst_root <ctrate_root> --device gpu
<TotalSegmentator env>/bin/python scripts/data/prepare_1mm.py --stage ts --dst_root <merlin_root> --device gpu
# HU / mask sanity check on a random sample (median HU in liver, spleen, lungs, vertebrae)
python scripts/data/prepare_1mm.py --stage check --dst_root <ctrate_root>
```

CT-RATE: if your copy stores raw values instead of HU, pass `--rescale_csv` with the CT-RATE metadata (columns
`VolumeName`, `RescaleSlope`, `RescaleIntercept`). The paper's CT-RATE and Merlin 1 mm volumes and masks came from an
earlier conversion with the same conventions, so voxel values can differ slightly from a fresh conversion. The paper's
CT-RATE harness numbers used 1 mm lung crops (HU clipped to [−1000, 1000]); the default export below uses the full
1 mm volumes (`scripts/data/export_audit_set.py --ctrate_src`).

## 2. Splits and manifests

```bash
python scripts/data/build_splits.py --env configs/env/local.yaml          # -> {work_root}/splits/{merlin,ctrate}_manifest.parquet
python scripts/data/prepare_manifests.py --env configs/env/local.yaml --merlin_test_labels merlin_test_labels.csv
python scripts/setup/verify_data.py --env configs/env/local.yaml           # every image / mask path exists
```

- **Merlin**: the official split (`meta_final.csv`); one study per patient.
- **CT-RATE**: the official `valid` split is the test set. Validation = 5% of training patients (seed 0). Every patient
  that appears in ReXGroundingCT val/test is held out (`rexgrounding_holdout`), so ReXGroundingCT stays an untouched
  lesion test for our models. The **v2 test set** drops the non-chest series listed in the CT-RATE data-correction note
  (`organic/data/ctrate_v2.py`; copy `dataset/metadata/no_chest_valid.txt` to `{work_root}/manifests/ctrate_v2/dataset/metadata/`).
- Volumes to exclude everywhere can be listed in `{work_root}/manifests/<dataset>_exclude.csv` (column `volume`).
- **Merlin test labels**: the paper's reference labels (5,109 test studies × 21 findings, re-extracted from the reports
  with LLMs; 0 / 1, empty = uncertain) are `protocol/files/merlin_test_labels.csv` of the benchmark
  (https://huggingface.co/datasets/zllhhhh/organic3d-benchmark). Without `--merlin_test_labels` the Merlin test labels
  stay empty and the Merlin test set cannot be scored. Your own labels can use the same layout or a long format
  (`patient_id`, `finding` = label column name in `organic/vocab.py` `FINDINGS`, `label`, `ground_truth`).

## 3. Report sentences (training text) and Merlin training labels

```bash
CUDA_VISIBLE_DEVICES=0 <vllm env>/bin/python scripts/data/parse_reports.py --env configs/env/local.yaml --shard 0/4   # one per GPU
<vllm env>/bin/python scripts/data/parse_reports.py --env configs/env/local.yaml --merge
python scripts/data/prepare_manifests.py --env configs/env/local.yaml      # re-run: fills Merlin train/val labels
```

Qwen3-14B (vLLM) splits each report into sentences and returns finding, polarity and organ per sentence, plus a negated
and an organ-swapped rewrite for stated abnormalities. Merlin training labels come from the rule layer
`organic/data/label_rules.py` on top of these claims (uncertain mentions are left unlabelled).

## 4. 2 mm training cache (optional, recommended)

```bash
python scripts/data/cache_volumes.py --env configs/env/local.yaml --workers 48
```

Anatomy-centred crop of 208³ voxels at 2 mm (the 192³ model input plus jitter room) per volume, image as int16 HU and
the 16-group organ map, in `{work_root}/cache/<dataset>/<volume>.npz`. Training and evaluation read the cache when
`data.use_cache` is true and fall back to the NIfTI otherwise.

## 5. External test sets

```bash
# RAD-ChestCT (Zenodo 6406114): convert the released volumes to NIfTI with their spacing, then
python scripts/data/prepare_1mm.py --stage resample --src_dir <RAD-ChestCT NIfTI> --dst_root <radchest_root> \
    --img_sub lung_nii/img --seg_sub lung_nii/seg
<TotalSegmentator env>/bin/python scripts/data/prepare_1mm.py --stage ts --dst_root <radchest_root> \
    --img_sub lung_nii/img --seg_sub lung_nii/seg
python scripts/data/prepare_radchest.py --env configs/env/local.yaml      # labels mapped to the 18 CT-RATE findings

# ReXGroundingCT (lesion masks of CT-RATE scans; place the release under <ctrate_root>/ReXGroundingCT)
python scripts/data/prepare_rexgrounding.py --env configs/env/local.yaml --workers 32

# AbdomenAtlas 3.0 (IID test split only; $AA3_ROOT = the download root)
AA3_ROOT=<AbdomenAtlas3.0> HF_TOKEN=<token> bash scripts/data/download_aa3_test.sh
AA3_ROOT=<AbdomenAtlas3.0> python scripts/data/prepare_aa3.py --stage resample --workers 16
AA3_ROOT=<AbdomenAtlas3.0> <TotalSegmentator env>/bin/python scripts/data/prepare_aa3.py --stage ts --device gpu
AA3_ROOT=<AbdomenAtlas3.0> python scripts/data/prepare_aa3.py --stage finalize --workers 16

# lesion-erasure (LE) and side sets on ReXGroundingCT and AbdomenAtlas 3.0
python scripts/data/prepare_heal.py --ds rex --workers 16
python scripts/data/prepare_heal.py --ds aa3 --workers 16
```

## 6. Benchmark sets for the harness

```bash
python scripts/data/export_audit_set.py --env configs/env/local.yaml --datasets ctrate,merlin,radchest
# ReXGroundingCT scans (exported only once every scan with a released mask is mapped; 3,041 scans; the paper's
# ctrate_rex_set_v2.parquet differs only in its `index` column)
python scripts/data/export_audit_set.py --env configs/env/local.yaml --datasets ctrate_rex --suffix _v2 --no_meta
python scripts/data/prepare_chest_set.py --env configs/env/local.yaml          # CT-RATE scans for the lung lobe test (LL)
```

This writes `{work_root}/extaudit/meta.json` (finding names, prompt phrases, erased organ groups as TotalSegmentator
ids, fills, noise) and one `<set>_set.parquet` per set (`index`, `volume`, `patient_id`, `label_*`, `img`, `seg`,
`lat_pairs` = the laterality sentence pairs). The LE sets are written by `prepare_heal.py`. Set
`EXTAUDIT_ROOT={work_root}/extaudit` for the harness wrappers.
