"""Config loading: base YAML + environment YAML (data roots) + ablation YAML + command-line dot-list overrides.

    cfg = load_config('configs/stageA.yaml', env='configs/env/local.yaml',
                      ablation='configs/encoder/enc4_organic3d.yaml', overrides=['train.max_steps=20'])

Every data path in the manifests is relative to a dataset root, so moving to another server only means a new
env file with different roots.
"""
import os
from omegaconf import OmegaConf


def load_config(base, env=None, ablation=None, overrides=None):
    parts = [OmegaConf.load(base)]
    for p in (env, ablation):
        if p:
            parts.append(OmegaConf.load(p))
    if overrides:
        parts.append(OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.merge(*parts)
    OmegaConf.resolve(cfg)
    return cfg


def dataset_root(cfg, dataset):
    if dataset == 'radchest':                       # RAD-ChestCT: external chest test set (evaluation only)
        return cfg.paths.radchest_root
    return {'merlin': cfg.paths.merlin_root, 'ctrate': cfg.paths.ctrate_root}[dataset]


def abspath(cfg, dataset, rel):
    """Manifest paths are '1mm/img/<id>.nii.gz' and '1mm/ts_seg/<id>.nii.gz'. A server whose files live elsewhere or in
    another format sets paths.layout.<dataset>.{img,seg} templates with {id}, e.g. for CT-RATE lung crops stored as .bin:
        layout: {ctrate: {img: gold/lung/img/{id}.bin, seg: gold/lung/seg/{id}.bin}}"""
    lay = (cfg.paths.get('layout') or {}).get(dataset) if 'paths' in cfg else None
    if lay:
        for kind, prefix in (('img', '1mm/img/'), ('seg', '1mm/ts_seg/')):
            if rel.startswith(prefix) and lay.get(kind):
                rid = rel[len(prefix):]
                rid = rid[:-7] if rid.endswith('.nii.gz') else rid
                return os.path.join(dataset_root(cfg, dataset), lay[kind].format(id=rid))
    return os.path.join(dataset_root(cfg, dataset), rel)


def read_manifest(cfg, dataset):
    """The dataset manifest minus excluded volumes. Exclusions: {work_root}/manifests/<dataset>_exclude.csv (column
    'volume'), e.g. CT-RATE volumes whose lung crop contains no lung (no valid chest evaluation). Every training,
    caching, evaluation and audit path reads manifests through here, so an exclusion applies everywhere at once."""
    import pandas as pd
    root = os.path.join(cfg.paths.work_root, 'manifests')
    m = pd.read_parquet(os.path.join(root, f'{dataset}.parquet'))
    ex = os.path.join(root, f'{dataset}_exclude.csv')
    if os.path.exists(ex):
        drop = set(pd.read_csv(ex).volume.astype(str))
        m = m[~m.volume.astype(str).isin(drop)]
    return m
