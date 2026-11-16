"""Ladder metrics for CT-CHAT from its per-scan P(Yes) shards ($CTCHAT_SCORES, glob; default
{work_root}/evalB/CT-CHAT/scores_shard*.npz: keys index, feat, P__<condition>, nvox__<condition>, written by a CT-CHAT
scorer run in CT-CHAT's own environment on the ctrate_audit1000 set, see extaudit/README.md), with the same metric code as NV-Reason-CT, MedGemma and our stage-B models. Writes {work_root}/evalB/CT-CHAT/.

    python scripts/ctchat_metrics.py --env configs/env/local.yaml
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from organic.config import load_config
from organic.audit_nvreason import audit_sample
from organic.evaluate_stageB import metrics
from organic.vocab import CTRATE_FINDINGS, FINDING_NAMES

ap = argparse.ArgumentParser(); ap.add_argument('--env', required=True); a = ap.parse_args()
cfg = load_config(os.path.join(os.path.dirname(__file__), '../configs/stageA.yaml'), env=a.env)
fs = sorted(glob.glob(os.environ.get('CTCHAT_SCORES', os.path.join(cfg.paths.work_root, 'evalB', 'CT-CHAT', 'scores_shard*.npz'))))
Z = [dict(np.load(f)) for f in fs]
cat = {k: np.concatenate([z[k] for z in Z]) for k in Z[0]}
o = np.argsort(cat['index']); cat = {k: v[o] for k, v in cat.items()}
n = len(cat['index']); print(len(fs), 'shards,', n, 'scans')
res = {k[3:]: torch.tensor(v) for k, v in cat.items() if k.startswith('P__')}
res['wrong'] = res['clean'][np.roll(np.arange(n), 1)]          # another patient's scan, same questions
res.update({'nvox_' + k[6:]: torch.tensor(v) for k, v in cat.items() if k.startswith('nvox__')})
res['feat'] = torch.tensor(cat['feat']); res['index'] = torch.tensor(cat['index'])
res['gen_index'] = []; res['texts'] = {'clean': [], 'wrong': []}; res['js'] = [FINDING_NAMES.index(f) for f in CTRATE_FINDINGS]
m = audit_sample(cfg, 'ctrate', 1000)
L = m[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32)
tab, summ = metrics(res, L, 'ctrate')
out = os.path.join(cfg.paths.work_root, 'evalB', 'CT-CHAT'); os.makedirs(out, exist_ok=True)
tab.to_csv(os.path.join(out, 'ctrate_ladder.csv'))
json.dump({'ctrate': summ}, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
print('SUMMARY', json.dumps({'run': 'CT-CHAT', 'dataset': 'ctrate', **summ}))
