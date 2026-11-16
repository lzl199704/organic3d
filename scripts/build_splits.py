"""Patient-level train/val/test manifests for Merlin and CT-RATE (1 mm NIfTI + TotalSegmentator v2 masks).

Merlin : official split from meta_final.csv (train / val / test).
CT-RATE: official train / valid split; 'valid' is the public test set. We carve a validation set from train by
         patient (5%, seed 0) and hold out every patient that appears in ReXGroundingCT val/test (drawn from the
         CT-RATE train split), so ReXGroundingCT can serve as an untouched lesion-localization test.

Output: <out>/{merlin,ctrate}_manifest.parquet (default <out> = {work_root}/splits, read by prepare_manifests.py)
Usage : python scripts/build_splits.py --env configs/env/local.yaml
        python scripts/build_splits.py --merlin_root <merlin> --ctrate_root <CT-RATE> --out <dir>
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default=None, help='env config (configs/env/local.yaml): default roots and <work_root>/splits')
    ap.add_argument('--merlin_root', default=None)
    ap.add_argument('--ctrate_root', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.env:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from organic.config import load_config
        cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../configs/stageA.yaml'), env=a.env)
        a.merlin_root = a.merlin_root or cfg.paths.merlin_root
        a.ctrate_root = a.ctrate_root or cfg.paths.ctrate_root
        a.out = a.out or os.path.join(cfg.paths.work_root, 'splits')
    assert a.merlin_root and a.ctrate_root and a.out, 'give --env or all of --merlin_root, --ctrate_root, --out'
    run(a.out, a.merlin_root.rstrip('/'), a.ctrate_root.rstrip('/'))


def run(OUT, MER, CTR):
    os.makedirs(OUT, exist_ok=True)

    # ---------------- Merlin
    m = pd.read_csv(f'{MER}/meta_final.csv').rename(columns={'study id': 'study_id', 'Findings': 'report', 'Split': 'split'})
    m['img'] = MER + '/1mm/img/' + m.study_id + '.nii.gz'
    m['seg'] = MER + '/1mm/ts_seg/' + m.study_id + '.nii.gz'
    m['exists'] = [os.path.exists(p) and os.path.exists(s) for p, s in zip(m.img, m.seg)]
    m['patient_id'] = m.study_id          # Merlin releases one study per patient
    m['dataset'] = 'merlin'
    m = m[['dataset', 'patient_id', 'study_id', 'split', 'img', 'seg', 'report', 'exists']]
    m.to_parquet(f'{OUT}/merlin_manifest.parquet', index=False)
    print('Merlin', m.groupby('split').agg(n=('study_id', 'size'), exists=('exists', 'sum')).to_dict('index'))

    # ---------------- CT-RATE
    vols = sorted(glob.glob(f'{CTR}/1mm/img/*/*/*.nii.gz'))
    c = pd.DataFrame({'img': vols})
    c['volume'] = c.img.str.rsplit('/', n=1).str[1]
    c['patient_id'] = c.img.str.split('/').str[-3]                     # e.g. train_7117
    c['scan_id'] = c.img.str.split('/').str[-2]                        # e.g. train_7117_b
    c['seg'] = c.img.str.replace('/1mm/img/', '/1mm/ts_seg/', regex=False)
    c['exists'] = [os.path.exists(s) for s in c.seg]
    c['official'] = c.patient_id.str.split('_').str[0]                 # train | valid
    rep = pd.concat([pd.read_csv(f'{CTR}/dataset/radiology_text_reports/{f}') for f in ['train_reports.csv', 'validation_reports.csv']])
    rep['report'] = ('Findings: ' + rep.Findings_EN.fillna('') + ' Impression: ' + rep.Impressions_EN.fillna('')).str.strip()
    lab = pd.concat([pd.read_csv(f'{CTR}/dataset/multi_abnormality_labels/{f}') for f in ['train_predicted_labels.csv', 'valid_predicted_labels.csv']])
    c = c.merge(rep[['VolumeName', 'report']], left_on='volume', right_on='VolumeName', how='left').drop(columns='VolumeName')
    c = c.merge(lab.rename(columns={'VolumeName': 'volume'}), on='volume', how='left')
    rex = json.load(open(f'{CTR}/ReXGroundingCT/dataset.json'))
    rex_hold = {x['name'].split('_')[0] + '_' + x['name'].split('_')[1] for k in ['val', 'test'] for x in rex[k]}
    rex_all = {x['name'] for k in rex for x in rex[k]}
    c['rexgrounding'] = c.volume.isin(rex_all)
    rng = np.random.default_rng(0)
    train_pat = sorted(set(c.patient_id[c.official == 'train']) - rex_hold)
    val_pat = set(rng.choice(train_pat, size=int(0.05 * len(train_pat)), replace=False))
    c['split'] = np.where(c.official == 'valid', 'test',
                 np.where(c.patient_id.isin(rex_hold), 'rexgrounding_holdout',
                 np.where(c.patient_id.isin(val_pat), 'val', 'train')))
    c['dataset'] = 'ctrate'
    c.to_parquet(f'{OUT}/ctrate_manifest.parquet', index=False)
    print('CT-RATE', c.groupby('split').agg(volumes=('volume', 'size'), patients=('patient_id', 'nunique'),
                                            report=('report', lambda s: int(s.notna().sum())), exists=('exists', 'sum')).to_dict('index'))


if __name__ == '__main__':
    main()
