"""Free-text findings reports from 3D CT VLMs (docs/metrics.md, "Report level"), each model in its own python env
(adapters in extaudit/vlm_models/), on the full test set (<dataset>_set.parquet). Clean scans only, greedy decoding,
each model's own report prompt (adapter.report_prompt). Scoring (one extractor per dataset for every model):
scripts/metrics/report_metrics_std.py.

Adapter API (in addition to the vlm_core API):
    report_prompt(dataset) -> str           the model's own report-generation request, as its authors use it
    report(hu, dataset, max_new_tokens) -> str   greedy report for the 1 mm RAS HU volume (cuda tensor)
    set_scan(seg), set_dataset(name)        optional, as in vlm_core

    CUDA_VISIBLE_DEVICES=0 <model env>/bin/python -m extaudit.vlm_report \
        --model omnict --name OmniCT-7B --dataset ctrate --root <work_root>/extaudit [--shard 0/2]
Writes {--out}/{name}/{dataset}_reports_shard{k}of{n}.jsonl (default --out = <work_root>/eval_reports, i.e. the folder
next to --root), one line per scan ({index, volume, prompt, text, sec}); a rerun skips scans already written (resume
after OOM / preemption).
"""
import argparse
import importlib
import json
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from extaudit.core import VolSet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='adapter module in extaudit/vlm_models')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--name', required=True)
    ap.add_argument('--dataset', default='ctrate')
    ap.add_argument('--root', required=True, help='{work_root}/extaudit')
    ap.add_argument('--n', type=int, default=0, help='first N scans of the set (0 = all)')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--max_new_tokens', type=int, default=768)
    ap.add_argument('--out', default=None, help='default: {root}/../eval_reports = {work_root}/eval_reports')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    k, n = map(int, a.shard.split('/'))
    if a.out is None:
        a.out = os.path.join(a.root, '..', 'eval_reports')
    rows = pd.read_parquet(os.path.join(a.root, f'{a.dataset}_set.parquet'))
    rows = (rows.head(a.n) if a.n else rows).iloc[k::n]
    out_dir = os.path.join(a.out, a.name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{a.dataset}_reports_shard{k}of{n}.jsonl')
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)['index'] for l in fh if l.strip()}
    rows = rows[~rows['index'].isin(done)]
    print(f'{a.name} {a.dataset} shard {k}/{n}: {len(rows)} scans to generate ({len(done)} already done)', flush=True)
    if not len(rows):
        return
    dev = torch.device('cuda')
    ad = importlib.import_module(f'extaudit.vlm_models.{a.model}').Adapter(a.ckpt, dev)
    if hasattr(ad, 'set_dataset'):
        ad.set_dataset(a.dataset)
    scan = getattr(ad, 'set_scan', None)
    prompt = ad.report_prompt(a.dataset)
    print('PROMPT', repr(prompt), flush=True)
    loader = DataLoader(VolSet(rows), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    t0, n_err = time.time(), 0
    with open(path, 'a') as fh:
        for it, b in enumerate(loader):
            if 'error' in b:
                n_err += 1; print('ERROR', b['index'], b['error'], flush=True)
                continue
            hu = b['hu'].to(dev)
            if scan:
                scan(b['seg'].to(dev))
            t1 = time.time()
            with torch.no_grad():
                text = ad.report(hu, a.dataset, a.max_new_tokens)
            rec = {'index': b['index'], 'volume': b['volume'], 'prompt': prompt, 'text': text,
                   'sec': round(time.time() - t1, 1)}
            rec.update(getattr(ad, 'last_meta', None) or {})      # e.g. our answer-then-write finding list
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            if it % 10 == 0:
                el = time.time() - t0
                print(f'{it + 1}/{len(rows)}  {el / 60:.1f} min, {el / (it + 1):.1f} s/scan | {text[:160]!r}', flush=True)
    print(f'DONE {a.name} {a.dataset} shard {k}/{n}: {len(rows) - n_err} reports, {n_err} errors, '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
