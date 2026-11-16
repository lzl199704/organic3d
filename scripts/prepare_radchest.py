"""Manifest for RAD-ChestCT (Draelos et al. 2021; Duke; public on Zenodo 6406114) as an external chest test set.

Volumes: 1 mm RAS lung crops with TotalSegmentator masks ({radchest_root}/lung_nii/{img,seg}/<id>.nii.gz;
HU clipped to [-1000, 1000], as the CT-RATE lung crops our models are evaluated on). Every scan is used as test
data (none of our models saw RAD-ChestCT); the dataset's own split is kept in `rad_subset`.

Labels: RAD-ChestCT's report-derived abnormality x location labels (84 x 52), mapped to the 18 CT-RATE findings below.
A finding is positive if any listed abnormality is positive at any listed location (None = any location).
Approximate mappings are marked; the 16-label subset used in the literature drops medical material and lymphadenopathy.

    python scripts/prepare_radchest.py --env configs/env/local.yaml
Writes {work_root}/manifests/radchest.parquet.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from organic.config import load_config
from organic.vocab import FINDING_NAMES

LUNG = ['left_upper', 'left_mid', 'left_lower', 'right_upper', 'right_mid', 'right_lower', 'right_lung', 'left_lung',
        'lung', 'interstitial', 'subpleural', 'centrilobular']
DEVICES = ['catheter_or_port', 'pacemaker_or_defib', 'stent', 'chest_tube', 'gi_tube', 'tracheal_tube',
           'heart_valve_replacement', 'hardware', 'clip', 'staple', 'suture']
MAP = {   # CT-RATE finding -> (RAD-ChestCT abnormalities, locations or None)
    'ct_medical_material': (DEVICES, None),
    'ct_arterial_calcification': (['atherosclerosis', 'calcification', 'plaque'], ['aorta', 'pulmonary_artery', 'svc']),
    'ct_cardiomegaly': (['cardiomegaly'], None),
    'ct_pericardial_effusion': (['pericardial_effusion'], None),
    'ct_coronary_calcification': (['coronary_artery_disease'], None),
    'ct_hiatal_hernia': (['hernia'], ['esophagus', 'stomach', 'diaphragm']),
    'ct_lymphadenopathy': (['lymphadenopathy'], None),
    'ct_emphysema': (['emphysema'], None),
    'ct_atelectasis': (['atelectasis'], None),
    'ct_lung_nodule': (['nodule', 'nodulegr1cm'], LUNG),
    'ct_lung_opacity': (['opacity', 'groundglass'], LUNG),
    'ct_fibrotic_sequela': (['fibrosis', 'scarring'], LUNG),
    'ct_pleural_effusion': (['pleural_effusion'], None),
    'ct_mosaic_attenuation': (['air_trapping'], None),            # approximate: no mosaic label in RAD-ChestCT
    'ct_peribronchial_thickening': (['bronchial_wall_thickening'], None),
    'ct_consolidation': (['consolidation'], None),
    'ct_bronchiectasis': (['bronchiectasis'], None),
    'ct_septal_thickening': (['septal_thickening'], None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../configs/stageA.yaml'), env=a.env)
    root = cfg.paths.radchest_root
    lab = pd.concat([pd.read_csv(f'{root}/data/img{s}_Abnormality_and_Location_Labels.csv').assign(rad_subset=s)
                     for s in ('train', 'valid', 'test')]).set_index('NoteAcc_DEID')
    summ = pd.read_csv(f'{root}/data/Summary_3630.csv').set_index('NoteAcc_DEID')
    have = {f[:-7] for f in os.listdir(f'{root}/lung_nii/img')} & {f[:-7] for f in os.listdir(f'{root}/lung_nii/seg')}
    lab = lab[lab.index.isin(have)]
    m = pd.DataFrame({'dataset': 'radchest', 'volume': lab.index, 'scan_id': lab.index,
                      'patient_id': summ.reindex(lab.index).MRN_DEID.fillna(pd.Series(lab.index, index=lab.index)).values,
                      'split': 'test', 'img_rel': [f'lung_nii/img/{v}.nii.gz' for v in lab.index],
                      'seg_rel': [f'lung_nii/seg/{v}.nii.gz' for v in lab.index], 'report': '',
                      'rad_subset': lab.rad_subset.values})
    for f in FINDING_NAMES:
        m[f'label_{f}'] = np.nan
    for f, (abns, locs) in MAP.items():
        cols = [c for c in lab.columns if '*' in c and c.split('*')[0] in abns and (locs is None or c.split('*')[1] in locs)]
        m[f'label_{f}'] = (lab[cols].max(axis=1) > 0).astype(float).values
    out = os.path.join(cfg.paths.work_root, 'manifests', 'radchest.parquet')
    m.to_parquet(out, index=False)
    print(f'wrote {out}: {len(m)} volumes, {m.patient_id.nunique()} patients, subsets {m.rad_subset.value_counts().to_dict()}')
    print(m[[f'label_{f}' for f in MAP]].sum().astype(int).rename(lambda c: c[9:]).to_string())


if __name__ == '__main__':
    main()
