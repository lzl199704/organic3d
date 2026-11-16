"""Write root-relative training manifests for Merlin and CT-RATE.

Input : the absolute-path split manifests from scripts/data/build_splits.py (default {work_root}/splits)
Output: {work_root}/manifests/{merlin,ctrate}.parquet with columns
        dataset, patient_id, scan_id, volume, split, img_rel, seg_rel, report, label_<finding>...
Labels: CT-RATE from the 18 released columns; Merlin train/val left NaN until report parsing fills them,
        Merlin test from --merlin_test_labels: the paper's reference labels, merlin_test_labels.csv of the benchmark
        (https://huggingface.co/datasets/zllhhhh/organic3d-benchmark, protocol/files/: study_id x 21 findings, 0 / 1,
        empty = uncertain), or your own labels in the same layout or in long format (patient_id, finding = label
        column name of vocab.FINDINGS, label, ground_truth; 'Uncertain' rows dropped). Without it they stay NaN.

Usage: python scripts/data/prepare_manifests.py --env configs/env/local.yaml [--src <dir>] [--merlin_test_labels <parquet>]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.config import load_config
from organic.vocab import FINDINGS, CTRATE_FINDINGS, MERLIN_FINDINGS

HERE = os.path.dirname(os.path.abspath(__file__))


def rel(paths, root):
    root = root.rstrip('/') + '/'
    bad = [p for p in paths if not p.startswith(root)]
    assert not bad, f'{len(bad)} paths outside {root}, e.g. {bad[:1]}'
    return [p[len(root):] for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--src', default=None, help='split manifests of build_splits.py (default {work_root}/splits)')
    ap.add_argument('--merlin_test_labels', default=None,
                    help="Merlin test reference labels: the benchmark's merlin_test_labels.csv (see docstring)")
    a = ap.parse_args()
    cfg = load_config(os.path.join(HERE, '../../configs/stageA.yaml'), env=a.env)
    a.src = a.src or os.path.join(cfg.paths.work_root, 'splits')
    out = os.path.join(cfg.paths.work_root, 'manifests')
    os.makedirs(out, exist_ok=True)

    m = pd.read_parquet(f'{a.src}/merlin_manifest.parquet')
    m['img_rel'] = rel(m.img, cfg.paths.merlin_root)
    m['seg_rel'] = rel(m.seg, cfg.paths.merlin_root)
    m['scan_id'] = m.study_id
    m['volume'] = m.study_id
    for f in FINDINGS:
        m[f'label_{f}'] = np.nan
    if a.merlin_test_labels and os.path.exists(a.merlin_test_labels):
        p = a.merlin_test_labels
        q = pd.read_csv(p) if p.endswith('.csv') else pd.read_parquet(p)
        test = m.split == 'test'
        if 'study_id' in q and any(f in q for f in MERLIN_FINDINGS):        # benchmark layout: study_id x finding keys
            w = q.set_index('study_id')
            for f in MERLIN_FINDINGS:
                if f in w:
                    m.loc[test, f'label_{f}'] = m.loc[test, 'study_id'].map(w[f]).values
        else:                                                                 # long format
            q = q[q.ground_truth != 'Uncertain']
            w = q.pivot_table(index='patient_id', columns='finding', values='label', aggfunc='max')
            for f in MERLIN_FINDINGS:
                col = FINDINGS[f][0]
                if col in w:
                    m.loc[test, f'label_{f}'] = m.loc[test, 'study_id'].map(w[col]).values
        print('merlin test labels from', p)
    ml_path = os.path.join(cfg.paths.work_root, 'parsed', 'merlin_labels.parquet')
    if os.path.exists(ml_path):                       # train/val labels parsed from reports (parse_reports.py --merge)
        ml = pd.read_parquet(ml_path).set_index('volume')
        tv = m.split.isin(['train', 'val'])
        for f in MERLIN_FINDINGS:
            if f'label_{f}' in ml:
                m.loc[tv, f'label_{f}'] = m.loc[tv, 'volume'].map(ml[f'label_{f}']).values
        print('merlin train/val labels from', ml_path)
    keep = ['dataset', 'patient_id', 'scan_id', 'volume', 'split', 'img_rel', 'seg_rel', 'report'] + [f'label_{f}' for f in FINDINGS]
    m[keep].to_parquet(f'{out}/merlin.parquet', index=False)

    c = pd.read_parquet(f'{a.src}/ctrate_manifest.parquet')
    c['img_rel'] = rel(c.img, cfg.paths.ctrate_root)
    c['seg_rel'] = rel(c.seg, cfg.paths.ctrate_root)
    for f in FINDINGS:
        c[f'label_{f}'] = c[FINDINGS[f][0]].astype(float) if f in CTRATE_FINDINGS else np.nan
    c[keep].to_parquet(f'{out}/ctrate.parquet', index=False)
    for name, d in [('merlin', m), ('ctrate', c)]:
        print(name, d.split.value_counts().to_dict())
    print('wrote', out)


if __name__ == '__main__':
    main()
