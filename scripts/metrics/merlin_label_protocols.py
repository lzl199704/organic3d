"""Merlin zero-shot AUROC under different test label sets (same per-scan scores, eval/<name>/merlin_scores.npz).

Label sets for the Merlin test scans:
  ref       the reference labels the harness scores against (manifests/merlin.parquet test rows; 'Uncertain'
            dropped). The paper's reference labels are NOT distributed: supply your own (scripts/data/prepare_manifests.py
            --merlin_test_labels).
  v4        our report labeller (organic/data/label_rules.py = frozen v4) on the test reports, --v4_labels (parquet, index =
            volume, columns label_mr_*; default {work_root}/manifests/merlin_test_labels_v4.parquet)
  file_drop Merlin's released zero_shot_findings_disease_cls.csv (--merlin_file; part of the Merlin data release) with -1 rows dropped (the protocol of the Merlin and
            RADAR zero-shot numbers; -1 hides most positives the file's rules miss)
  file_neg  the same file with -1 counted as negative
Scored on the findings the file shares with our 21 (SHARED) and, for ref / v4, on all findings with >= 10 positives.
Also writes per-finding label agreement (positives per source, share of v4 positives the file codes -1).

    python scripts/metrics/merlin_label_protocols.py --env configs/env/local.yaml --merlin_file <zero_shot_findings_disease_cls.csv> \
        [--v4_labels <parquet>] [--names enc4_organic3d,COLIPRI,...]
Writes {work_root}/eval/merlin_label_protocols.csv and merlin_label_agreement.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.config import load_config

SHARED = {'mr_hydronephrosis': 'hydronephrosis', 'mr_gallstones': 'gallstones',
          'mr_biliary_dilatation': 'biliary_ductal_dilation', 'mr_hepatomegaly': 'hepatomegaly',
          'mr_fatty_liver': 'hepatic_steatosis', 'mr_hiatal_hernia': 'hiatal_hernia',
          'mr_pancreatic_atrophy': 'pancreatic_atrophy', 'mr_splenomegaly': 'splenomegaly',
          'mr_appendicitis': 'appendicitis', 'mr_bowel_obstruction': 'bowel_obstruction',
          'mr_aortic_aneurysm': 'abdominal_aortic_aneurysm', 'mr_prostatomegaly': 'prostatomegaly',
          'mr_ascites': 'ascites', 'mr_lymphadenopathy': 'lymphadenopathy'}
NAMES = 'enc4_organic3d,COLIPRI,Jolia,Jolia-long,SPECTRE,Merlin'


def macro(score, lab, cols, min_pos=10):
    out = {}
    for f in cols:
        y, s = lab[f], score[f]
        ok = ~(np.isnan(y) | np.isnan(s))
        if ok.sum() and min_pos <= y[ok].sum() <= ok.sum() - min_pos:
            out[f] = roc_auc_score(y[ok], s[ok])
    return float(np.mean(list(out.values()))) if out else np.nan, len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--names', default=NAMES)
    ap.add_argument('--merlin_file', required=True, help="Merlin's released zero_shot_findings_disease_cls.csv")
    ap.add_argument('--v4_labels', default=None, help='default {work_root}/manifests/merlin_test_labels_v4.parquet')
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env)
    wr = cfg.paths.work_root
    meta = json.load(open(os.path.join(wr, 'extaudit', 'meta.json')))
    fi = {f: j for j, f in enumerate(meta['findings'])}
    mr = [f for f in meta['findings'] if f.startswith('mr_')]
    v4 = pd.read_parquet(a.v4_labels or os.path.join(wr, 'manifests', 'merlin_test_labels_v4.parquet'))
    v4.columns = [c.replace('label_', '') for c in v4.columns]
    fl = pd.read_csv(a.merlin_file).set_index('study_id')

    rows, agree = [], None
    for name in a.names.split(','):
        p = os.path.join(wr, 'eval', name, 'merlin_scores.npz')
        if not os.path.exists(p):
            print(f'{name}: no merlin_scores.npz, skipped')
            continue
        z = np.load(p, allow_pickle=True)
        vol = pd.Index(z['patient'].astype(str))
        score = {f: z['delta__clean'][:, fi[f]].astype(float) for f in mr}
        L = {'ref': {f: z['labels'][:, fi[f]].astype(float) for f in mr},
             'v4': {f: v4.reindex(vol)[f].to_numpy(float) for f in mr}}
        F = fl.reindex(vol)
        L['file_drop'] = {f: F[c].where(F[c] >= 0).to_numpy(float) for f, c in SHARED.items()}
        L['file_neg'] = {f: F[c].clip(lower=0).to_numpy(float) for f, c in SHARED.items()}
        r = {'model': name, 'n': len(vol)}
        for k in ('file_drop', 'file_neg', 'ref', 'v4'):
            r[f'{k}_shared'], r[f'{k}_shared_nf'] = macro(score, L[k], list(SHARED))
        for k in ('ref', 'v4'):
            r[f'{k}_all'], r[f'{k}_all_nf'] = macro(score, L[k], mr)
        rows.append(r)
        if agree is None:                      # label statistics do not depend on the model
            agree = pd.DataFrame([{
                'finding': f, 'file_col': c,
                'pos_file': int(np.nansum(L['file_drop'][f])),
                'minus1_file': int(np.isnan(L['file_drop'][f]).sum() - F[c].isna().sum()),
                'pos_ref': int(np.nansum(L['ref'][f])), 'pos_v4': int(np.nansum(L['v4'][f])),
                'v4_pos_coded_minus1': float(np.mean(F[c].to_numpy()[L['v4'][f] == 1] == -1)),
                'v4_pos_found_by_file': float(np.mean(F[c].to_numpy()[L['v4'][f] == 1] == 1)),
                'ref_v4_agree': float(np.nanmean(L['ref'][f] == L['v4'][f])),
            } for f, c in SHARED.items()])
    t = pd.DataFrame(rows)
    os.makedirs(os.path.join(wr, 'eval'), exist_ok=True)
    t.to_csv(os.path.join(wr, 'eval', 'merlin_label_protocols.csv'), index=False)
    agree.to_csv(os.path.join(wr, 'eval', 'merlin_label_agreement.csv'), index=False)
    pd.set_option('display.width', 200)
    print(t.round(3).to_string(index=False))
    print(agree.round(3).to_string(index=False))
    print('totals: file positives', agree.pos_file.sum(), '| v4 positives', agree.pos_v4.sum(),
          '| v4 positives the file codes -1', f'{np.average(agree.v4_pos_coded_minus1, weights=agree.pos_v4):.3f}',
          '| found by file', f'{np.average(agree.v4_pos_found_by_file, weights=agree.pos_v4):.3f}')


if __name__ == '__main__':
    main()
