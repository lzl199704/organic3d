"""CPU-only preflight before any GPU job: everything except GPU speed.

  1. manifests and dataset roots
  2. volume cache complete (same volume list as cache_volumes.py) and readable (random sample)
  3. models load offline from HF_HOME (COLIPRI, Qwen3.5-4B tokenizer/config, NV-Reason-CT processor)
  4. one real training batch: dataset -> sampler pairs -> text batch -> counterfactual plan -> Organic3D forward on CPU

    HF_HUB_OFFLINE=1 python scripts/setup/preflight_cpu.py --env configs/env/local.yaml
Prints PREFLIGHT OK or the failed checks, and writes {work_root}/logs/preflight.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.config import abspath, load_config, read_manifest

HERE = os.path.dirname(os.path.abspath(__file__))
res, fails = {}, []


def check(name, fn):
    t = time.time()
    try:
        out = fn()
        res[name] = {'ok': True, 'detail': out, 'sec': round(time.time() - t, 1)}
        print(f'[ok]   {name}: {out}', flush=True)
    except Exception as e:
        res[name] = {'ok': False, 'detail': f'{type(e).__name__}: {e}'}
        fails.append(name)
        print(f'[FAIL] {name}: {type(e).__name__}: {e}', flush=True)


def expected_cache(cfg):
    out = []
    for d in ('ctrate', 'merlin'):
        m = read_manifest(cfg, d)
        m = m[m.split.isin(['train', 'val', 'test'])]
        if d == 'ctrate':
            m = pd.concat([m[m.split == 'test'], m[m.split != 'test'].groupby('scan_id', group_keys=False).head(1)])
        out += [os.path.join(cfg.paths.work_root, 'cache', d, f'{v}.npz') for v in m.volume]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--sample', type=int, default=40)
    a = ap.parse_args()
    cfg = load_config(os.path.join(HERE, '../../configs/stageA.yaml'), env=a.env)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    W = cfg.paths.work_root

    def manifests():
        # a dataset without raw files here is fine when it is read only through the cache (e.g. a cache built on
        # another machine and copied); check 2 then verifies that cache is complete
        info = {}
        for d in ('merlin', 'ctrate'):
            m = read_manifest(cfg, d)
            r = m.iloc[0]
            raw = all(os.path.exists(abspath(cfg, d, r[k])) for k in ('img_rel', 'seg_rel'))
            if not raw:
                assert os.path.exists(os.path.join(W, 'cache', d, f'{r.volume}.npz')), \
                    f'{d}: neither raw files ({abspath(cfg, d, r.img_rel)}) nor a cache file'
            info[d] = {**m.split.value_counts().to_dict(), 'source': 'raw + cache' if raw else 'cache only'}
        return info
    check('1 manifests and roots', manifests)

    def cache():
        exp = expected_cache(cfg)
        have = [p for p in exp if os.path.exists(p)]
        assert len(have) == len(exp), f'{len(exp) - len(have)} of {len(exp)} cache files missing (cache job not finished?)'
        rng = np.random.default_rng(0)
        for p in rng.choice(have, size=min(a.sample, len(have)), replace=False):
            z = np.load(p)
            assert z['img'].dtype == np.int16 and z['img'].ndim == 3 and z['organ'].shape == z['img'].shape, p
        gb = sum(os.path.getsize(p) for p in rng.choice(have, size=200)) / 200 * len(have) / 1e9
        return f'{len(have)} files, ~{gb:.0f} GB, {a.sample} random files readable'
    check('2 volume cache', cache)

    def models():
        from colipri import get_model, get_processor
        get_model(pretrained=True); get_processor()
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B'); AutoConfig.from_pretrained('Qwen/Qwen3.5-4B')
        AutoProcessor.from_pretrained('nvidia/NV-Reason-CT', trust_remote_code=True)
        from huggingface_hub import snapshot_download
        for r in ('Qwen/Qwen3.5-4B', 'nvidia/NV-Reason-CT'):
            p = snapshot_download(r, local_files_only=True)
            assert any(f.endswith('.safetensors') for f in os.listdir(p)), f'{r}: weights missing'
        return 'COLIPRI, Qwen3.5-4B, NV-Reason-CT load from the local cache'
    check('3 models offline', models)

    def batch():
        import torch
        torch.set_num_threads(min(16, os.cpu_count() or 4))
        from colipri import get_processor
        from organic.augment import build_image_batch, plan_counterfactuals
        from organic.data import CTDataset, MixedBatchSampler, collate, load_manifests, load_sentences
        from organic.model import build_model
        from organic.train_stageA import build_text
        c = load_config(os.path.join(HERE, '../../configs/stageA.yaml'), env=a.env, overrides=['data.limit=400'])
        sents = load_sentences(c)
        ds = CTDataset(c, load_manifests(c, 'train'), train=True, sentences=sents)
        smp = MixedBatchSampler(ds, 2, dict(c.data.dataset_weights), 1.0, 0, 1)
        idx = next(iter(smp))
        b = collate([ds[i] for i in idx])
        rng = np.random.default_rng(0)
        txt = build_text(b, c, rng)
        plans = plan_counterfactuals(b, c, rng)
        x, _ = build_image_batch(b, plans, [], c, torch.device('cpu'), seed=0)
        model = build_model(c).eval()
        with torch.no_grad():
            T, A, _ = model.encode_image(x[:1].float())
        return (f'{len(sents)} claim sets, {sum(len(v) for v in smp.pairs.values())} matched pairs in 400-row slice, '
                f'{len(txt["pos"])} sentences, {sum(p is not None for p in plans)} counterfactual plans, tokens {tuple(T.shape)}')
    check('4 one batch through the model (CPU)', batch)

    os.makedirs(os.path.join(W, 'logs'), exist_ok=True)
    with open(os.path.join(W, 'logs', 'preflight.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('PREFLIGHT OK' if not fails else f'PREFLIGHT FAILED: {fails}', flush=True)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
