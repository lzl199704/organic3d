"""Convert this repository's per-scan outputs into the score-file schema of the public benchmark metric
(Hugging Face benchmark package, metric/organic3d_benchmark.py, `SCHEMA`; metric card "Score files"). CPU only.

The harness and the stage-A/B evaluators keep their own formats (all analysis scripts here read them); this script is
the bridge to the public metric. It is the converter used to recompute the public leaderboard from the paper's outputs.

What it maps (internal -> public):
  encoder  {W}/eval/<name>/<dataset>_scores.npz      index -> scan_id (via {W}/extaudit/<dataset>_set.parquet);
           delta__<cond> -> score__<cond> (first 18 columns = the 18 chest findings, public order);
           nvox__rem_* kept; feat kept; lat (per-scan counts [index, n, ok_clean, ok_mirror, both]) ->
           side_scan_id / side_scores: per side sentence, indicator scores with the same joint counts (the harness
           stores counts, not sentence scores, so side correct / side follows image are exact but the scores are 0/1)
  vlm      {W}/extaudit/out_vlm/<name>/<dataset>_shard*.npz (D__ or P__ keys) or <folder>/<dataset>_scores.npz of the
           NV-Reason-CT / MedGemma audits -> score__clean, score__mirror, score__rem_<group>@fat, nvox__..., feat
  grounding {W}/extaudit/out/<name>/grounding_shard*.parquet (+ grounding_v2new_shard*, or grounding_v2_shard* for a
           model run on the whole v2 set) -> scan_id, k, s_clean, s_les, s_ctrl, ctrl_overlap, ctrl_in_lung,
           d_clean / d_les / d_ctrl (18 chest findings)
With --refs it also writes reference files WITH labels (refs/*.parquet) for local checks; never distribute those
(the public references are built by the benchmark's protocol/builders from your own dataset copies).

    python scripts/export_hf_scores.py --out <dir> [--models enc4_organic3d,COLIPRI,...] [--only enc,grounding,vlm]
        [--refs] [--vlm_audit NV-Reason-CT=<folder>,MedGemma-1.5-4B=<folder>]
W = work root (scripts/workroot.py: $ORGANIC3D_WORK_ROOT, else paths.work_root of $ORGANIC3D_ENV).
Writes <out>/scores/<name>__<dataset>__<task>.{npz,parquet}.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from organic.ctrate_v2 import no_chest_volumes  # noqa: E402
from workroot import work_root  # noqa: E402

W = work_root()


def chest_cols():
    """The 18 CT-RATE findings: the first 18 of the internal vocabulary, same order as the public one."""
    f = json.load(open(os.path.join(W, 'extaudit', 'meta.json')))['findings']
    cols = [i for i, x in enumerate(f) if x.startswith('ct_')]
    assert cols == list(range(18))
    return [f[i] for i in cols]


def sid(v):
    return str(v).replace('.nii.gz', '')


# ------------------------------------------------------------------ references (local checks only)
def ref_encoder(d):
    st = pd.read_parquet(f'{W}/extaudit/{d}_set.parquet')
    if d == 'ctrate':
        st = st[~st.volume.isin(no_chest_volumes(W))]
    r = pd.DataFrame({'scan_id': st.volume.map(sid).values, 'patient_id': st.patient_id.astype(str).values,
                      'order': st['index'].values})
    for f in chest_cols():
        r[f'label_{f}'] = st[f'label_{f}'].astype(float).values
    return r


def ref_vlm(d):
    st = pd.read_parquet(f'{W}/extaudit/{d}_audit1000_set.parquet')
    r = pd.DataFrame({'scan_id': st.volume.map(sid).values, 'patient_id': st.patient_id.astype(str).values,
                      'order': st['index'].values})
    for f in chest_cols():
        r[f'label_{f}'] = st[f'label_{f}'].astype(float).values
    return r


def ref_rex():
    gt = pd.read_parquet(f'{W}/rexground/findings.parquet')
    return pd.DataFrame({'scan_id': gt.volume.map(sid), 'k': gt.k.astype(int), 'patient_id': gt.patient.astype(str),
                         'category': gt.category, 'subset': np.where(gt.our_split.isin(['rexgrounding_holdout', 'test']),
                                                                     'held_out', 'training_scans')})


# ------------------------------------------------------------------ encoder
def lat_to_sentences(lat, idx2sid):
    """Per-scan counts [index, n, ok_clean, ok_mirror, both] -> per-sentence indicator scores [true clean, swapped
    clean, true mirrored, swapped mirrored] with the same joint counts."""
    sids, sc = [], []
    for vi, n, o0, o1, b in lat:
        n, o0, o1, b = int(round(n)), int(round(o0)), int(round(o1)), int(round(b))
        s = idx2sid[int(vi)]
        for cnt, row in ((b, (1, 0, 0, 1)), (o0 - b, (1, 0, 1, 0)), (o1 - b, (0, 1, 0, 1)), (n - o0 - o1 + b, (0, 1, 1, 0))):
            assert cnt >= 0
            sids += [s] * cnt
            sc += [row] * cnt
    return np.asarray(sids, dtype=str), np.asarray(sc, np.float32).reshape(-1, 4)


def conv_encoder(model, d, out):
    f = f'{W}/eval/{model}/{d}_scores.npz'
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    st = pd.read_parquet(f'{W}/extaudit/{d}_set.parquet').set_index('index')
    idx = z['index']
    o = {'scan_id': np.asarray([sid(v) for v in st.loc[idx, 'volume']], dtype=str),
         'findings': np.asarray(chest_cols(), dtype=str)}
    for k in z.files:
        if k.startswith('delta__'):
            o['score__' + k[7:]] = z[k][:, :18].astype(np.float32)
        elif k.startswith('nvox__') and k not in ('nvox__clean', 'nvox__mirror'):
            o[k] = z[k].astype(np.int64)
    if 'feat' in z.files:
        o['feat'] = z['feat']
    lat = z['lat'] if 'lat' in z.files else []
    if len(lat):
        o['side_scan_id'], o['side_scores'] = lat_to_sentences(lat, dict(zip(idx.tolist(), o['scan_id'].tolist())))
    p = f'{out}/scores/{model}__{d}__encoder.npz'
    np.savez_compressed(p, **o)
    return p


# ------------------------------------------------------------------ VLM
def vlm_from_harness(name, d):
    fs = sorted(glob.glob(f'{W}/extaudit/out_vlm/{name}/{d}_shard*.npz'))
    if not fs:
        return None
    Z = [dict(np.load(f)) for f in fs]
    key = 'D__' if any(k.startswith('D__') for k in Z[0]) else 'P__'
    cat = {k: np.concatenate([z[k] for z in Z]) for k in Z[0] if k.startswith((key, 'nvox__', 'index', 'feat'))}
    o = np.argsort(cat['index'])
    cat = {k: v[o] for k, v in cat.items()}
    S = {k[3:]: v for k, v in cat.items() if k.startswith(key)}
    NV = {k[6:]: v for k, v in cat.items() if k.startswith('nvox__') and k[6:].startswith('rem_')}
    return cat['index'], [str(x) for x in Z[0]['findings']], S, NV, cat['feat']


def vlm_from_audit(folder, d):
    """<folder>/<d>_scores.npz of organic/audit_nvreason.py or audit_medgemma.py."""
    f = f'{folder}/{d}_scores.npz'
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    S = {('clean' if k == 'clean' else 'mirror' if k == 'mirror' else k + '@fat'): z[k] for k in z.files
         if k in ('clean', 'mirror') or k.startswith('rem_')}
    NV = {k[5:] + '@fat': z[k] for k in z.files if k.startswith('nvox_rem_')}
    return z['index'], chest_cols(), S, NV, z['feat']


def conv_vlm(model, d, out, audits):
    src = vlm_from_audit(audits[model], d) if model in audits else vlm_from_harness(model, d)
    if src is None:
        return None
    idx, finds, S, NV, feat = src
    st = pd.read_parquet(f'{W}/extaudit/{d}_audit1000_set.parquet').set_index('index')
    o = {'scan_id': np.asarray([sid(v) for v in st.loc[idx, 'volume']], dtype=str), 'findings': np.asarray(finds, dtype=str),
         'feat': np.asarray(feat, np.float32)}
    for k, v in S.items():
        o['score__' + k] = np.asarray(v, np.float64)[:, :len(finds)]
    for k, v in NV.items():
        o['nvox__' + k] = np.asarray(v, np.int64)
    p = f'{out}/scores/{model}__{d}__vlm.npz'
    np.savez_compressed(p, **o)
    return p


# ------------------------------------------------------------------ lesions
def conv_grounding(model, out):
    """ReXGroundingCT v2: old shards + the v2new shards; a model without old shards uses its full v2 run (same file
    selection as the paper's v2 set)."""
    old = sorted(glob.glob(f'{W}/extaudit/out/{model}/grounding_shard*.parquet'))
    fs = old + sorted(glob.glob(f'{W}/extaudit/out/{model}/grounding_v2new_shard*.parquet')) if old else \
        sorted(glob.glob(f'{W}/extaudit/out/{model}/grounding_v2_shard*.parquet'))
    if not fs:
        return None
    t = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    assert not t.duplicated(['volume', 'k']).any(), model
    o = pd.DataFrame({'scan_id': t.volume.map(sid), 'k': t.k.astype(int), 's_clean': t.s_clean, 's_les': t.s_les,
                      's_ctrl': t.s_ctrl, 'ctrl_overlap': t.ctrl_overlap, 'ctrl_in_lung': t.ctrl_in_lung})
    for c in ('d_clean', 'd_les', 'd_ctrl'):
        o[c] = [list(np.asarray(x, float)[:18]) for x in t[c]]
    p = f'{out}/scores/{model}__rexgroundingct__grounding.parquet'
    o.to_parquet(p, index=False)
    return p


ENCODERS = ['enc4_organic3d', 'CT-CLIP', 'CT-VocabFine', 'fVLM', 'ViSD-Boost', 'HLIP', 'ItemizedCLIP', 'COLIPRI',
            'Jolia', 'Jolia-long', 'SPECTRE', 'Merlin', 'RADAR-plus', 'RADAR-plus-ft']
VLMS = ['vlm4_organic3d', 'vlm1_plain', 'Hulu-Med-7B', 'OmniCT-7B', 'M3D-LaMed', 'RadFM',
        'NV-Reason-CT', 'MedGemma-1.5-4B']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default='', help='comma-separated names (default: all known)')
    ap.add_argument('--only', default='enc,grounding,vlm')
    ap.add_argument('--refs', action='store_true', help='also write labelled reference files (local checks only)')
    ap.add_argument('--vlm_audit', default='', help='name=folder of a <dataset>_scores.npz audit output, comma-separated')
    a = ap.parse_args()
    os.makedirs(f'{a.out}/scores', exist_ok=True)
    parts = a.only.split(',')
    if a.refs:
        os.makedirs(f'{a.out}/refs', exist_ok=True)
        for d in ('ctrate', 'radchest'):
            ref_encoder(d).to_parquet(f'{a.out}/refs/{d}_test_labelled.parquet', index=False)
        ref_vlm('ctrate').to_parquet(f'{a.out}/refs/ctrate_vlm1000_labelled.parquet', index=False)
        ref_rex().to_parquet(f'{a.out}/refs/rexgroundingct_findings.parquet', index=False)
    audits = dict(x.split('=', 1) for x in a.vlm_audit.split(',') if x)
    names = [m for m in a.models.split(',') if m]
    for m in names or ENCODERS:
        if 'enc' in parts:
            for d in ('ctrate', 'radchest'):
                print(m, d, conv_encoder(m, d, a.out))
        if 'grounding' in parts:
            print(m, 'grounding', conv_grounding(m, a.out))
    if 'vlm' in parts:
        for m in names or VLMS + list(audits):
            print(m, 'vlm', conv_vlm(m, 'ctrate', a.out, audits))


if __name__ == '__main__':
    main()
