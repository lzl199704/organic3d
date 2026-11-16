"""Aggregate scripts/slot_attention_maps.py --mode stats into the slot-analysis tables (aggregate numbers only; Merlin
contributes no scan-level output here).

    python scripts/summarise_slot_attention.py --src <stats dir> --out <assets dir>
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True, help='--out of slot_attention_maps.py --mode stats')
ap.add_argument('--out', required=True, help='folder for the aggregate tables')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
N = 24 ** 3

o = pd.read_csv(os.path.join(a.src, 'per_scan_organ.csv'))
o['lift'] = o.own / o.share
org = (o.groupby(['model', 'organ'])
        .agg(n_scans=('volume', 'size'), datasets=('dataset', lambda s: '+'.join(sorted(set(s)))),
             mean_tokens=('n_tok', 'mean'), token_share=('share', 'mean'), own_slots=('own', 'mean'),
             other_organ_slots=('other', 'mean'), global_slots=('glob', 'mean'), own_lift=('lift', 'median'))
        .reset_index())
org.to_csv(os.path.join(a.out, 'slot_attention_per_organ.csv'), index=False, float_format='%.4f')
by_ds = (o.groupby(['model', 'dataset'])[['share', 'own', 'other', 'glob']].mean()
         .rename(columns={'share': 'token_share', 'own': 'own_slots', 'other': 'other_organ_slots', 'glob': 'global_slots'}))

e = pd.read_csv(os.path.join(a.src, 'per_scan_entropy.csv'))
ent = e.groupby(['model', 'dataset'])[['eff_organ', 'eff_global', 'eff_organ_median', 'eff_global_median']].mean()

b = pd.read_csv(os.path.join(a.src, 'per_scan_beta.csv'))
bf = (b.groupby(['model', 'dataset', 'finding', 'organs'])[['beta_organ', 'beta_global', 'uniform']].mean().reset_index())
bf.to_csv(os.path.join(a.out, 'slot_attention_beta_per_finding.csv'), index=False, float_format='%.4f')
bd = b.groupby(['model', 'dataset'])[['beta_organ', 'beta_global', 'uniform']].mean()

for f in os.listdir(a.src):
    if f.startswith('lift_') and f.endswith('.csv'):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f'slot_attention_{f}'))


def nest(df):
    out = {}
    for idx, row in df.iterrows():
        d = out
        for key in (idx if isinstance(idx, tuple) else (idx,))[:-1]:
            d = d.setdefault(str(key), {})
        d[str(idx[-1] if isinstance(idx, tuple) else idx)] = {c: round(float(v), 4) for c, v in row.items()}
    return out


summary = {
    'definitions': {
        'own_slots': "mean over an organ group's 8 slots of their last-layer attention mass on tokens whose 8^3 patch "
                     "holds any voxel of that organ (TotalSegmentator, 2 mm crop; losses.anatomy_loss)",
        'other_organ_slots': "mean over the 120 slots of the other 15 organ groups of their mass on the same tokens",
        'global_slots': "mean over the 16 global slots of their mass on the same tokens",
        'token_share': "the organ's share of the 13,824 tokens = mass of a uniform slot",
        'eff_tokens': "exp(entropy) of a slot's attention over 13,824 tokens",
        'beta_organ': "sentence pooling weight of the finding's positive canonical prompt on the finding's organ slots; "
                      "uniform = 8 |O(f)| / 144",
    },
    'scans': {m: {d: int(n) for d, n in g.groupby('dataset').volume.nunique().items()} for m, g in o.groupby('model')},
    'per_dataset_mass': nest(by_ds),
    'eff_tokens': nest(ent),
    'beta': nest(bd),
}
json.dump(summary, open(os.path.join(a.out, 'slot_attention_summary.json'), 'w'), indent=1)
pd.set_option('display.width', 200)
print(by_ds.round(3)); print(ent.round(1)); print(bd.round(3))
print(org[org.model == 'enc4_organic3d'].round(3).to_string(index=False))
print(org[org.model != 'enc4_organic3d'].round(3).to_string(index=False))
