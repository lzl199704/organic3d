"""GREEN (Ostmeier et al., Findings of EMNLP 2024; StanfordAIMI/GREEN-radllama2-7b judge, RadEval's local implementation)
for the full-test-set reports of Table 3, sharded so it can run on several GPUs.

Same report/reference pairs as scripts/metrics/report_metrics_std.py (its load() and references(); CT-RATE v2 drops the 3
non-chest series): per model and dataset the reports are split into --shard k/n by position and scored one GPU per
process; per-report GREEN goes to {W}/eval/green/<name>__<dataset>__shard<k>of<n>.csv. --aggregate merges complete
shard sets and writes the mean into {W}/eval/report_std.csv (column 'green'; other columns untouched).

--max_length (default 4096, the judge's Llama-2 context): RadEval's default 2,048-token window covers prompt plus the
judge's output, and a batch is padded to its longest prompt. CT report pairs reach 1,000-1,900 prompt tokens (Merlin
median ~1,400), so at 2,048 the judge's analysis is often cut off before its summary and parsed as GREEN 0. Items are
scored in order of length so batches pad less.

    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/report_green.py --names vlm4_organic3d --datasets ctrate --shard 0/4
    python scripts/metrics/report_green.py --aggregate
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from report_metrics_std import W, load, references  # noqa: E402

OUT = f'{W}/eval/green'


def pairs(name, d):
    st = pd.read_parquet(f'{W}/extaudit/{d}_set.parquet')
    if d == 'ctrate':
        from organic.data.ctrate_v2 import no_chest_volumes
        st = st[~st.volume.isin(no_chest_volumes(W))]
    rep = load(name, d)
    if not st['index'].isin(rep).all():
        raise SystemExit(f'{name} {d}: incomplete reports')
    refs, _, _ = references(d, st.volume)
    return st['index'].to_list(), [rep[int(i)] or ' ' for i in st['index']], list(refs)


def score(names, datasets, shard, batch, ml):
    from RadEval.metrics.green_score.green import GREEN
    k, n = map(int, shard.split('/'))
    g = GREEN('StanfordAIMI/GREEN-radllama2-7b', output_dir=OUT, num_gpus=1)
    g.batch_size = batch                               # RadEval default 8
    g.max_length = ml                                  # RadEval default 2048 (prompt + output)
    os.makedirs(OUT, exist_ok=True)
    for name in names.split(','):
        for d in datasets.split(','):
            dst = f'{OUT}/{name}__{d}__shard{k}of{n}.csv'
            if os.path.exists(dst):
                continue
            idx, hyps, refs = pairs(name, d)
            sel = list(range(k, len(idx), n))
            sel.sort(key=lambda i: len(refs[i]) + len(hyps[i]))
            mean, std, per, _ = g([refs[i] for i in sel], [hyps[i] for i in sel])
            pd.DataFrame({'index': [idx[i] for i in sel], 'green': list(per)}).to_csv(dst, index=False)
            print(f'{name} {d} shard {k}/{n}: {len(sel)} reports, mean GREEN {mean:.4f}', flush=True)


def aggregate():
    rows = {}
    for f in glob.glob(f'{OUT}/*__shard*of*.csv'):
        name, d, sh = os.path.basename(f)[:-4].split('__')
        n = int(sh.split('of')[1])
        rows.setdefault((name, d, n), []).append(pd.read_csv(f))
    std = pd.read_csv(f'{W}/eval/report_std.csv')
    std['green'] = float('nan')                       # complete shard sets only
    for (name, d, n), parts in rows.items():
        if len(parts) != n:
            print(f'{name} {d}: {len(parts)} / {n} shards, not merged'); continue
        t = pd.concat(parts)
        m = (std.model == name) & (std.dataset == d)
        std.loc[m, 'green'] = float(t.green.mean())
        print(f'{name} {d}: n={len(t)} GREEN {t.green.mean():.4f}')
    std.to_csv(f'{W}/eval/report_std.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', default='')
    ap.add_argument('--datasets', default='ctrate,merlin')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--max_length', type=int, default=4096)
    a = ap.parse_args()
    aggregate() if a.aggregate else score(a.names, a.datasets, a.shard, a.batch, a.max_length)


if __name__ == '__main__':
    main()
