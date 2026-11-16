"""Scan list for the harness chest rung (extaudit/chest.py, --task chest; 2026-10-24): the CT-RATE v2 test scans of
extaudit/ctrate_set.parquet (img, seg, labels) plus, per scan, the lobe sentences of its report chosen exactly as
organic/evaluate_chest.py does (lobe_sentences on the manifest report with split_sentences): present statements that
name one lung lobe once, with its side, as a template with a {LOBE} slot, the true lobe index (LOBES order) and the
finding word. No image data is written; the harness reads the same img / seg files as the ladder.

    python scripts/data/prepare_chest_set.py --env configs/env/local.yaml
Writes {work_root}/extaudit/ctrate_chest_set.parquet (column lobe_items = JSON list of [template, lobe, word]).
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.config import load_config, read_manifest          # noqa: E402
from organic.ctrate_v2 import no_chest_volumes                  # noqa: E402
from organic.data import split_sentences                        # noqa: E402
from organic.evaluate_chest import lobe_sentences               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(here, '../../configs/stageA.yaml'), env=a.env, overrides=['paths.layout=null'])
    W = cfg.paths.work_root
    st = pd.read_parquet(os.path.join(W, 'extaudit', 'ctrate_set.parquet'))
    st = st[~st.volume.isin(no_chest_volumes(W))].copy()
    m = read_manifest(cfg, 'ctrate')
    rep = m[m.split == 'test'].set_index('volume').report
    assert st.volume.isin(rep.index).all()
    st['lobe_items'] = [json.dumps(lobe_sentences(rep[v], split_sentences)) for v in st.volume]
    n_items = st.lobe_items.map(lambda s: len(json.loads(s)))
    out = os.path.join(W, 'extaudit', 'ctrate_chest_set.parquet')
    st.to_parquet(out, index=False)
    print(f'{len(st)} scans, {int((n_items > 0).sum())} with lobe sentences, {int(n_items.sum())} sentences -> {out}')


if __name__ == '__main__':
    main()
