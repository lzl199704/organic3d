"""Zero-shot tumour detection on AbdomenAtlas 3.0 (2026-10-27). Scores, for every test (all AUROC; higher = tumour present):
  det            each encoder's own prompt template and score rule, exactly as the MLC columns (Eq. 1;
                 extaudit/native_prompts.py): positive minus negative prompt, which is monotone in each model's own
                 two-prompt softmax (CT-CLIP, Merlin, HLIP, fVLM, ViSD-Boost, RADAR+ ensembles), equals ItemizedCLIP's
                 positive-only score (null negative) and Jolia / SPECTRE's template-mean rule  -> Table 1
  native_single  the native positive prompt alone
  free           one sentence pair for every encoder, "There is a mass in the X." minus "There is no mass in the X."
  single         the free positive sentence alone
Tests:
  <organ>_tumour_aa3  AbdomenAtlas 3.0 IID test (extaudit/aa3_det.py -> aa3_det_shard*; labels scripts/data/prepare_aa3.py),
                      liver / pancreas / kidney per scan: positive = a lesion >= 10 mm, negative = no lesion in the organ
                      (radiologist-revised masks); organs with only < 10 mm lesions left out
Patient-level bootstrap 95% CIs (2,000; AbdomenAtlas: scan = patient).
    python scripts/metrics/lesion_det_metrics.py [--names enc4_organic3d,...]
Writes {W}/eval/lesion_det_summary.csv.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesion_cls_metrics import ENCODERS, R, W, auroc, boot  # noqa: E402


def shards(name, tag):
    fs = sorted(glob.glob(f'{R}/out/{name}/{tag}_shard*.parquet'))
    if not fs:
        return None
    ns = {int(f.rsplit('of', 1)[1].split('.')[0]) for f in fs}
    if len(ns) != 1 or len(fs) != ns.pop():
        print(f'{name}: incomplete {tag} shards'); return None
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', default=','.join(ENCODERS))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    aa3 = pd.read_parquet(f'{W}/aa3/cases.parquet') if os.path.exists(f'{W}/aa3/cases.parquet') else None
    out = []
    for n in a.names.split(','):
        t = shards(n, 'aa3_det')                     # AbdomenAtlas 3.0 IID test: checked negatives (scan = patient)
        if t is not None:
            t = t.merge(aa3, on='case')
            for organ in ('liver', 'pancreas', 'kidney'):
                u = t[t[f'{organ}_label'].isin(['positive', 'negative'])]
                y, g = (u[f'{organ}_label'] == 'positive').to_numpy(), u.case.astype(str).to_numpy()
                for kind, s in (('det', (u[f'n_{organ}'] - u[f'n_{organ}_no']).to_numpy()),     # native template
                                ('native_single', u[f'n_{organ}'].to_numpy()),
                                ('free', (u[f's_{organ}'] - u[f's_{organ}_no']).to_numpy()),    # free sentences
                                ('single', u[f's_{organ}'].to_numpy())):
                    lo, hi = boot(lambda r: auroc(y[r], s[r]), g, rng)
                    out.append(dict(model=n, test=f'{organ}_tumour_aa3', score=kind, n=len(u), n_pos=int(y.sum()),
                                    value=auroc(y, s), lo=lo, hi=hi))
    o = pd.DataFrame(out)
    dst = f'{W}/eval/lesion_det_summary.csv'
    if os.path.exists(dst):                      # keep other models' rows
        old = pd.read_csv(dst)
        o = pd.concat([old[~old.model.isin(set(o.model))], o], ignore_index=True)
    o.to_csv(dst, index=False)
    pd.set_option('display.width', 200)
    print(o[o.score == 'det'].pivot_table(index='model', columns='test', values='value').round(3).T.to_string())


if __name__ == '__main__':
    main()
