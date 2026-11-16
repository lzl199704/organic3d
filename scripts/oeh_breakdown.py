"""OEH (Table 2) and its parts, for the trained fat fill and the unseen fills (supplement S8).

For each VLM harness run (extaudit/out_vlm/<name>/<dataset>_shard*.npz, audit set) and dataset:
  oeh             = target withdrawal - control withdrawal (= report_tracks 'retraction', the Table 2 OEH), 95% CI
  wd_target       share of correctly stated findings withdrawn after erasing the finding's organ
  wd_control      share withdrawn after erasing a control organ (false retractions)
  fp_target       share of correctly negated findings that turn to "Yes" after erasing the finding's organ
  p_clean_pos     mean P(Yes) of correctly stated findings on the clean scan
  p_target_pos    mean P(Yes) of the same findings after their organ is erased
Answers use the Table 2 rule (report_tracks: per-finding cross-fitted thresholds on the clean scan). Size findings are
skipped as in the OEH test. A run name '<model>__<fill>' holds one unseen fill (extaudit/vlm_core.py --fills).

    python scripts/oeh_breakdown.py --names vlm4_organic3d,vlm4_organic3d__median,...
Writes {W}/eval/oeh_breakdown.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import report_tracks as rt                                   # noqa: E402
from organic.vocab import FINDING_NAMES, SIZE_FINDINGS, control_organ_ids, finding_organ_ids   # noqa: E402


def parts(src, d):
    P, nv, finds, L, vol = src
    if d == 'ctrate':
        k = ~np.isin(vol, list(rt.no_chest_volumes(rt.W)))
        P, nv = {c: v[k] for c, v in P.items()}, {c: v[k] for c, v in nv.items()}
        L, vol = L[k], vol[k]
    cols = [FINDING_NAMES.index(f) for f in finds]
    Y = L[:, cols]
    pats = np.array([rt.patient(v, d) for v in vol])
    T = rt.crossfit(P['clean'], Y, pats)
    stated = (P['clean'] >= T) & ~np.isnan(T)
    sig = lambda x: 1 / (1 + np.exp(-x)) if np.nanmax(np.abs(x)) > 1 else x     # D (logit difference) -> P(Yes)
    tg, ct, fp, pc, pt = [], [], [], [], []
    for c, Pc in P.items():
        if not c.startswith('rem_'):
            continue
        ids = set(rt.organ_ids(c[4:].split('+') if '+' in c else c[4:]))
        present = nv.get(c, np.ones(len(Pc))) > 0
        for j, f in enumerate(finds):
            if f in SIZE_FINDINGS:
                continue
            tgt, ctl = set(finding_organ_ids(f)), set(control_organ_ids(f))
            is_t = bool(tgt and ids & tgt)
            if not is_t and not ids <= ctl:
                continue
            ok = ~np.isnan(T[:, j]) & present
            pos = ok & stated[:, j] & (Y[:, j] == 1)
            neg = ok & ~stated[:, j] & (Y[:, j] == 0)
            if pos.sum() >= 5:
                (tg if is_t else ct).append((Pc[pos, j] < T[pos, j]).mean())
                if is_t:
                    pc.append(sig(P['clean'][pos, j]).mean()); pt.append(sig(Pc[pos, j]).mean())
            if is_t and neg.sum() >= 5:
                fp.append((Pc[neg, j] >= T[neg, j]).mean())
    m = lambda x: float(np.mean(x)) if x else np.nan
    return dict(wd_target=m(tg), wd_control=m(ct), fp_target=m(fp), p_clean_pos=m(pc), p_target_pos=m(pt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', required=True)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--first', type=int, default=0, help='score only scans that every listed run has (subset runs)')
    a = ap.parse_args()
    rows = []
    for d in ('ctrate', 'merlin'):
        srcs = {n: rt.from_harness(n, d) for n in a.names.split(',')}
        srcs = {n: s for n, s in srcs.items() if s is not None}
        if not srcs:
            continue
        # common scans: subset runs (--first) cover fewer scans than the full audit runs they are compared with
        common = set.intersection(*[set(map(str, s[4])) for s in srcs.values()])
        for n, (P, nv, finds, L, vol) in srcs.items():
            k = np.isin(vol.astype(str), list(common))
            s = ({c: v[k] for c, v in P.items()}, {c: v[k] for c, v in nv.items()}, finds, L[k], vol[k])
            ev = rt.evaluate(s, d, a.boot, np.random.default_rng(0))
            rows.append(dict(model=n.split('__')[0], fill=(n.split('__')[1] if '__' in n else 'fat'), dataset=d,
                             n=int(k.sum()), oeh=ev['retraction'], oeh_lo=ev['retraction_lo'],
                             oeh_hi=ev['retraction_hi'], **parts(s, d)))
    t = pd.DataFrame(rows)
    out = os.path.join(rt.W, 'eval', 'oeh_breakdown.csv')
    t.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    print(t.round(3).to_string(index=False))
    print('wrote', out)


if __name__ == '__main__':
    main()
