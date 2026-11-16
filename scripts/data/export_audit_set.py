"""Export a test set for the benchmark harness (extaudit/): one row per test volume, in evaluate_stageA's order,
so `index` joins our models' *_scores.npz. Also writes the shared metadata (finding names, prompt phrases, erased organ
groups as TotalSegmentator ids) that the model adapters read, so they need no organic import.

    python scripts/data/export_audit_set.py --env configs/env/local.yaml --datasets ctrate,merlin,radchest
    python scripts/data/export_audit_set.py --env configs/env/local.yaml --datasets ctrate_rex --suffix _v2 --no_meta
Datasets: ctrate, merlin, radchest (the full test sets; encoders and VLMs), ctrate_rex (ReXGroundingCT scans; run after
prepare_rexgrounding.py; the harness reads it as ctrate_rex_set_v2.parquet).
Writes {work_root}/extaudit/{dataset}_set{suffix}.parquet and {work_root}/extaudit/meta.json.

ctrate_rex refuses to export from an incomplete rexground/findings.parquet (every ReXGroundingCT scan with a released
mask and a volume in our manifest must have been mapped; re-run prepare_rexgrounding.py --only_missing first).
--keep <set>: rows of that file are copied verbatim (same `index`); scans not in it are appended with new indices.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.config import abspath, load_config, read_manifest
from organic.evaluate_stageA import EVAL_ORGANS, cond_name, lateral_sentences
from organic.vocab import FINDING_NAMES, FINDINGS, MERLIN_FINDINGS, ORGANS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--datasets', default='ctrate,merlin')
    ap.add_argument('--ctrate_src', default='1mm', choices=['gold', '1mm'],
                    help="CT-RATE test volumes: the full 1 mm NIfTI (default, public data) or 'gold' = 1 mm lung crops "
                         "(HU clipped to [-1000, 1000]) in <ctrate_root>/lung_nii, which is what the paper's harness numbers used")
    ap.add_argument('--suffix', default='', help='write {dataset}_set{suffix}.parquet (e.g. _v2)')
    ap.add_argument('--keep', default=None, help='existing set whose rows are kept verbatim; only new volumes appended')
    ap.add_argument('--no_meta', action='store_true', help='do not rewrite meta.json')
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env)
    out = os.path.join(cfg.paths.work_root, 'extaudit')
    os.makedirs(out, exist_ok=True)
    for d in a.datasets.split(','):
        if d == 'ctrate_rex':                               # ReXGroundingCT scans (any split), for --task grounding
            gt = pd.read_parquet(os.path.join(cfg.paths.work_root, 'rexground', 'findings.parquet'))
            m = read_manifest(cfg, 'ctrate')
            from prepare_rexgrounding import REX                # completeness check (see docstring)
            rex = json.load(open(os.path.join(REX, 'dataset.json')))
            have = set(os.listdir(os.path.join(REX, 'segmentations')))
            exp = {x['name'] for s in rex for x in rex[s] if x['name'] in have} & set(m.volume)
            mapped = {f.replace('.npz', '.nii.gz') for f in os.listdir(os.path.join(cfg.paths.work_root, 'rexground', 'masks'))}
            miss = sorted(exp - mapped)
            assert not miss, f'{len(miss)} ReXGroundingCT scans not mapped yet (re-run prepare_rexgrounding.py --only_missing): {miss[:5]}'
            print(f'ctrate_rex: {len(exp)} masked scans in manifest, all mapped; {gt.volume.nunique()} with >= 1 finding')
            m = m[m.volume.isin(set(gt.volume))]
        else:
            m = read_manifest(cfg, d)
            m = m[m.split == 'test']
        if d == 'merlin':                                   # same filter as evaluate_stageA.run
            m = m[m[[f'label_{f}' for f in MERLIN_FINDINGS]].notna().any(axis=1)]
        m = m.reset_index(drop=True)
        t = m[['volume', 'patient_id'] + [f'label_{f}' for f in FINDING_NAMES]].copy()
        t.insert(0, 'index', range(len(m)))
        src = 'ctrate' if d == 'ctrate_rex' else d.split('_')[0]
        t['img'] = [abspath(cfg, src, r) for r in m.img_rel]
        t['seg'] = [abspath(cfg, src, r) for r in m.seg_rel]
        if d == 'ctrate' and a.ctrate_src == 'gold':     # the lung crops = what our models were evaluated on
            g = os.path.join(cfg.paths.ctrate_root, 'lung_nii')
            t['img'] = [f'{g}/bin/{r[8:-7]}.bin' for r in m.img_rel]          # '1mm/img/<id>.nii.gz'
            t['seg'] = [f'{g}/seg/{r[11:]}' for r in m.seg_rel]               # '1mm/ts_seg/<id>.nii.gz'
            assert all(os.path.exists(x) for x in t.img.head(50)) and all(os.path.exists(x) for x in t.seg.head(50))
        t['lat_pairs'] = [json.dumps(lateral_sentences(r)) for r in m.report]
        if a.keep:                                          # old rows verbatim (same index), new volumes appended
            old = pd.read_parquet(a.keep)
            assert list(old.columns) == list(t.columns), 'column mismatch with --keep set'
            new = t[~t.volume.isin(set(old.volume))].copy()
            new['index'] = range(int(old['index'].max()) + 1, int(old['index'].max()) + 1 + len(new))
            print(f'{d}: kept {len(old)} rows of {a.keep}, appended {len(new)} new volumes')
            t = pd.concat([old, new.astype(old.dtypes.to_dict())], ignore_index=True)
        dst = os.path.join(out, f'{d}_set{a.suffix}.parquet')
        assert a.suffix or not a.keep, 'use --suffix with --keep (never overwrite the kept set)'
        t.to_parquet(dst, index=False)
        print(d, len(t), 'volumes,', int((t.lat_pairs != '[]').sum()), 'with side sentences ->', dst)
    if a.no_meta:
        return
    meta = {'findings': FINDING_NAMES, 'label_name': {f: FINDINGS[f][0] for f in FINDING_NAMES},
            'phrase': {f: FINDINGS[f][2] for f in FINDING_NAMES},
            'eval_organs': {d: [[cond_name(o), sorted(sum((ORGANS[g] for g in (o if isinstance(o, list) else [o])), []))]
                                for o in EVAL_ORGANS[d]] for d in EVAL_ORGANS},
            'fills_hu': {'fat': -100.0, 'soft': 40.0}, 'noise_sd': 15.0}
    with open(os.path.join(out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
