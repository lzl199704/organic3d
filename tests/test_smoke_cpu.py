"""CPU smoke test on synthetic data: no GPU, no downloads, no real CT, no report text.

Covers the pieces every result depends on:
  * vocabulary / organ routing, counterfactual planning and image edits (organic.augment)
  * every stage-A loss (finite values, gradients flow)
  * anatomy slots: a slot depends only on its own query and the image (the property the slot swap relies on)
  * AUROC and matched-pair AUROC against brute-force references
  * the benchmark harness end to end: synthetic 1 mm NIfTI scans + TotalSegmentator-style masks -> extaudit.core
    ladder (clean, mirror, organ erasure with two fills) with a toy encoder -> organic.evaluate_stageA metrics. The toy
    encoder reads only lung density, so erasing the lungs must remove its emphysema signal while erasing control organs
    must not (evidence rung), which checks the intervention, bookkeeping and metric code together.

    python tests/test_smoke_cpu.py          (or: python -m pytest tests/)
"""
import argparse
import json
import os
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''                     # never touch a GPU
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from organic.config import load_config  # noqa: E402
from organic.vocab import (FINDING_NAMES, FINDINGS, ORGAN_ID, ORGAN_NAMES, ORGANS, CTRATE_FINDINGS,  # noqa: E402
                            control_organ_ids, finding_organ_ids)

CFG = os.path.join(ROOT, 'configs', 'stageA.yaml')


# ----------------------------------------------------------------------------------------------- synthetic data
def synthetic_chest(shape=(64, 64, 64), lung_hu=-850.0, seed=0):
    """A toy 1 mm RAS 'chest': soft-tissue body, two lungs (left = small x), heart, aorta, trachea, oesophagus.
    Returns HU float32 and a uint8 mask with TotalSegmentator v2 ids."""
    rng = np.random.default_rng(seed)
    X, Y, Z = np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')
    hu = np.full(shape, -1000.0, np.float32)
    seg = np.zeros(shape, np.uint8)
    body = ((X - 32) / 30.0) ** 2 + ((Y - 32) / 26.0) ** 2 <= 1
    hu[body] = 40.0
    for x0, tid in ((18, 10), (46, 12)):                    # left lung (upper lobe id 10), right lung (id 12)
        lung = ((X - x0) / 11.0) ** 2 + ((Y - 32) / 16.0) ** 2 + ((Z - 32) / 26.0) ** 2 <= 1
        hu[lung] = lung_hu
        seg[lung] = tid
    for (x0, y0, r, tid, v) in ((32, 24, 6, 51, 45.0), (32, 40, 3, 52, 200.0), (32, 44, 2, 15, 30.0)):
        m = ((X - x0) ** 2 + (Y - y0) ** 2 <= r * r) & (abs(Z - 32) <= 12)
        hu[m] = v
        seg[m] = tid
    trachea = ((X - 32) ** 2 + (Y - 32) ** 2 <= 4) & (Z > 44)
    hu[trachea] = -1000.0
    seg[trachea] = 16
    hu += rng.normal(0, 10, shape).astype(np.float32)
    return hu, seg


def _meta():
    """= scripts/data/export_audit_set.py meta.json (finding names, prompt phrases, erased organ groups as TS ids)."""
    from organic.evaluate_stageA import EVAL_ORGANS, cond_name
    return {'findings': FINDING_NAMES, 'label_name': {f: FINDINGS[f][0] for f in FINDING_NAMES},
            'phrase': {f: FINDINGS[f][2] for f in FINDING_NAMES},
            'eval_organs': {d: [[cond_name(o), sorted(sum((ORGANS[g] for g in (o if isinstance(o, list) else [o])), []))]
                                for o in EVAL_ORGANS[d]] for d in EVAL_ORGANS},
            'fills_hu': {'fat': -100.0, 'soft': 40.0}, 'noise_sd': 15.0}


class LungDensityAdapter:
    """Toy encoder in the harness adapter API: its only feature is the mean lung density, and only the emphysema
    prompt reads it (Delta = score(positive) - score(negative) = -mean lung HU / 1000)."""
    lung_ids = [10, 11, 12, 13, 14]

    def __init__(self, ckpt=None, device='cpu'):
        self.lung = None

    @staticmethod
    def prompts(f, meta):
        p = meta['phrase'][f]
        return f'There is {p}.', f'No {p}.'

    def encode_text(self, sents):
        return torch.tensor([[1.0] if s == 'There is emphysema.' else [0.0] for s in sents])

    def set_scan(self, seg):
        self.lung = torch.isin(seg, torch.as_tensor(self.lung_ids, dtype=seg.dtype))

    def encode_image(self, hu):
        z = torch.stack([-hu[self.lung].mean() / 1000.0])
        return z, z


# ----------------------------------------------------------------------------------------------- tests
def test_vocab_and_routing():
    from organic.routing import route_matrix, sentence_organs
    assert len(FINDING_NAMES) == 39 and len(ORGAN_NAMES) == 16
    for f in FINDING_NAMES:                                  # targets and controls never overlap
        assert not set(finding_organ_ids(f)) & set(control_organ_ids(f)), f
    assert sentence_organs('There is a mass in the left kidney.') == [ORGAN_ID['kidney_left']]
    assert sentence_organs('There is hydronephrosis.') == sorted([ORGAN_ID['kidney_left'], ORGAN_ID['kidney_right']])
    assert sentence_organs('No medical material.') == []
    oq = torch.cat([torch.arange(128) // 8, torch.full((16,), -1)])
    R = route_matrix(['There is a mass in the left kidney.', 'No medical material.'], oq)
    assert R.shape == (2, 144) and R[0].sum() == 8 and R[1].sum() == 16 and R[1, 128:].all()


def test_counterfactual_batch():
    from organic.augment import build_image_batch, lateral_swap, organ_mask, plan_counterfactuals, severity_swap
    cfg = load_config(CFG, overrides=['aug.ev_prob=1.0'])
    rng = np.random.default_rng(0)
    organs = []
    for s in range(2):
        hu, seg = synthetic_chest(seed=s)
        from organic.data import _LUT
        organs.append(torch.from_numpy(_LUT[seg]))
    labels = torch.full((2, len(FINDING_NAMES)), float('nan'))
    labels[:, [FINDING_NAMES.index(f) for f in CTRATE_FINDINGS]] = 0.0
    labels[0, FINDING_NAMES.index('ct_emphysema')] = 1.0
    hu = torch.stack([torch.from_numpy(synthetic_chest(seed=s)[0]) for s in range(2)])[:, None]
    batch = {'labels': labels, 'organ': torch.stack(organs), 'img': hu}
    plans = plan_counterfactuals(batch, cfg, rng)
    assert any(p is not None for p in plans)
    x, idx = build_image_batch(batch, plans, [0], cfg, torch.device('cpu'), seed=0)
    assert x.shape[1:] == hu.shape[1:] and float(x.abs().max()) <= 1.0
    for i, p in enumerate(plans):
        if p is None:
            continue
        clean = x[i, 0]
        if i in idx['sens']:                                 # edits stay inside the dilated target + U_t mask
            m = organ_mask(batch['organ'][i], p['target'] + p['U_t'], cfg.aug.dilate)
            changed = (x[idx['sens'][i], 0] != clean)
            assert not changed[~m].any() and changed[m].float().mean() > 0.5
        if i in idx['inv']:
            m = organ_mask(batch['organ'][i], p['U'], cfg.aug.dilate)
            assert not (x[idx['inv'][i], 0] != clean)[~m].any()
    assert torch.equal(x[idx['mirror'][0], 0], torch.flip(x[0, 0], dims=[0]))
    assert lateral_swap('Nodule in the left lower lobe.') == 'Nodule in the right lower lobe.'
    assert severity_swap('Mild small bowel dilatation.') == 'Severe small bowel dilatation.'


def test_losses_backward():
    from organic import losses as L
    torch.manual_seed(0)
    V, S, F_ = 4, 6, len(FINDING_NAMES)
    Smat = torch.randn(V, S, requires_grad=True)
    pos = torch.zeros(V, S, dtype=torch.bool); pos[torch.arange(V), torch.arange(V)] = True
    true = pos.clone(); true[0, 5] = True
    delta = torch.randn(V, F_, requires_grad=True)
    labels = torch.full((V, F_), float('nan')); labels[:, :3] = torch.tensor([[1., 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]])
    b = torch.zeros(F_, requires_grad=True)
    A = torch.softmax(torch.randn(1, 144, 6 ** 3, requires_grad=True), -1)
    organ = torch.full((1, 48, 48, 48), 255, dtype=torch.uint8); organ[0, :24] = ORGAN_ID['lung_left']; organ[0, 24:] = ORGAN_ID['lung_right']
    oq = torch.cat([torch.arange(128) // 8, torch.full((16,), -1)])
    terms = {
        'i2t': L.i2t_loss(Smat, pos, true), 't2i': L.t2i_loss(Smat, pos, true, 0.5),
        'txt': L.txt_loss(Smat[:, 0], torch.cat([Smat[:, 1:3], torch.full((V, 1), float('-inf'))], 1)),
        'cls': L.cls_loss(delta, labels, b),
        'ev': L.evidence_loss(delta[:2, 0], delta[2:, 0], delta[:2, 1], torch.tensor([0, 0]), torch.tensor([1, 1]), b, 1.0),
        'pair': L.pair_loss(delta, labels, 1.0), 'mir': L.mirror_loss(Smat[:, 0], Smat[:, 1]),
        'anat': L.anatomy_loss(A, organ, oq, len(ORGAN_NAMES)),
    }
    total = sum(terms.values())
    total.backward()
    assert all(torch.isfinite(v) for v in terms.values()), terms
    assert Smat.grad.abs().sum() > 0 and delta.grad.abs().sum() > 0


def test_anatomy_slots_are_independent():
    """Slots never attend to each other: changing one slot's query changes only that slot's output."""
    from organic.model import AnatomyQueries, grid_positions
    torch.manual_seed(0)
    q = AnatomyQueries(kv_dim=32, d=48, n_organs=16, q_per_organ=8, q_global=16, layers=2, heads=4, use_pe=False).eval()
    H, pos = torch.randn(1, 24 ** 3, 32), grid_positions('cpu')
    with torch.no_grad():
        T0, A, pbar = q(H, pos)
        q.queries[5] += 1.0
        T1, _, _ = q(H, pos)
    assert T0.shape == (1, 144, 48) and A.shape == (1, 144, 24 ** 3) and pbar.shape == (1, 144, 3)
    diff = (T1 - T0).abs().sum(-1)[0]
    assert diff[5] > 1e-3 and torch.all(diff[torch.arange(144) != 5] < 1e-5)
    assert torch.allclose(A.sum(-1), torch.ones(1, 144), atol=1e-5)
    assert q.organ_of_query()[:8].tolist() == [0] * 8 and q.organ_of_query()[-1] == -1


def test_auroc_and_matched_pairs():
    from sklearn.metrics import roc_auc_score
    from organic.ladder import matched_pair_auroc
    from organic.train_stageA import auroc
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(float)
    s = np.round(y + rng.normal(0, 1, 200), 1)              # with ties
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-12
    L = rng.integers(0, 2, (200, 3)).astype(float); L[rng.random((200, 3)) < 0.1] = np.nan
    j = 0
    got, n = matched_pair_auroc(L, s, j)
    num = den = 0                                            # brute force over all positive/negative pairs
    ok = ~np.isnan(L[:, j])
    for a in np.where(ok & (L[:, j] == 1))[0]:
        for b in np.where(ok & (L[:, j] == 0))[0]:
            if np.array_equal(np.nan_to_num(L[a, 1:], nan=-1), np.nan_to_num(L[b, 1:], nan=-1)):
                num += (s[a] > s[b]) + 0.5 * (s[a] == s[b]); den += 1
    assert n == den and abs(got - num / den) < 1e-12


def test_harness_ladder_end_to_end():
    import nibabel as nib
    import pandas as pd
    from extaudit import core
    from organic.evaluate_stageA import metrics
    meta = _meta()
    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        n = 16
        for i in range(n):
            emph = i % 2 == 0                                # emphysema scans: darker lungs
            hu, seg = synthetic_chest(lung_hu=-950.0 if emph else -850.0, seed=i)
            img, msk = os.path.join(tmp, f'vol{i}.nii.gz'), os.path.join(tmp, f'seg{i}.nii.gz')
            nib.save(nib.Nifti1Image(hu, np.eye(4)), img)
            nib.save(nib.Nifti1Image(seg, np.eye(4)), msk)
            r = {'index': i, 'volume': f'vol{i}.nii.gz', 'patient_id': f'p{i}', 'img': img, 'seg': msk,
                 'lat_pairs': json.dumps([['Opacity in the left lung.', 'Opacity in the right lung.']])}
            for f in FINDING_NAMES:
                r[f'label_{f}'] = (float(emph) if f == 'ct_emphysema' else float(i % 4 == 1)) if f in CTRATE_FINDINGS else np.nan
            rows.append(r)
        pd.DataFrame(rows).to_parquet(os.path.join(tmp, 'ctrate_set.parquet'), index=False)
        a = argparse.Namespace(dataset='ctrate', fills='fat,soft', workers=1, root=tmp, name='toy')
        mdl = core.Model(LungDensityAdapter(), meta, 'ctrate')
        done, nerr, _ = core.run_ladder(a, meta, mdl, pd.DataFrame(rows), torch.device('cpu'), 0, 1)
        assert done == n and nerr == 0
        z = np.load(os.path.join(tmp, 'out', 'toy', 'ctrate_shard0of1.npz'))
        conds = list(z['conds'])
        assert conds[:2] == ['clean', 'mirror'] and len(conds) == 2 + 2 * len(meta['eval_organs']['ctrate'])
        res = {c: {'delta': torch.from_numpy(z[f'delta__{c}']), 'feat': torch.from_numpy(z[f'feat__{c}'].astype(np.float32)),
                   'nvox': torch.from_numpy(z[f'nvox__{c}'])} for c in conds}
        lab = pd.DataFrame(rows).set_index('index').loc[z['index'], [f'label_{f}' for f in FINDING_NAMES]].to_numpy(float)
        df = metrics(res, lab, 'ctrate', ['fat', 'soft'], min_pos=4)
    e = df.loc['ct_emphysema']
    assert e.zero_shot == 1.0                                 # the toy feature separates the classes
    assert e.effect_fat < -0.1 and e.effect_soft < -0.1       # lung erasure removes the evidence, controls do not
    assert abs(e.matched_pair_auroc - 1.0) < 1e-9



def test_harness_chest_task():
    """--task chest (extaudit/chest.py): density edits and lobe erasure run end to end; the emphysema edit lowers the
    toy encoder's emphysema score on emphysema scans and moves no other finding (metric code of evaluate_chest)."""
    import nibabel as nib
    import pandas as pd
    from extaudit import core
    from extaudit.chest import run_chest
    from organic.evaluate_chest import metrics_edits
    meta = _meta()
    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        for i in range(8):
            emph = i % 2 == 0
            hu, seg = synthetic_chest(lung_hu=-990.0 if emph else -850.0, seed=i)
            img, msk = os.path.join(tmp, f'vol{i}.nii.gz'), os.path.join(tmp, f'seg{i}.nii.gz')
            nib.save(nib.Nifti1Image(hu, np.eye(4)), img)
            nib.save(nib.Nifti1Image(seg, np.eye(4)), msk)
            r = {'index': i, 'volume': f'vol{i}.nii.gz', 'patient_id': f'p{i}', 'img': img, 'seg': msk,
                 'lobe_items': json.dumps([['Opacity in the {LOBE}.', 3, 'opacity']] if i < 4 else [])}
            for f in FINDING_NAMES:
                r[f'label_{f}'] = (float(emph) if f == 'ct_emphysema' else float(i % 4 == 1)) if f in CTRATE_FINDINGS else np.nan
            rows.append(r)
        pd.DataFrame(rows).to_parquet(os.path.join(tmp, 'ctrate_chest_set.parquet'), index=False)
        a = argparse.Namespace(workers=1, root=tmp, name='toy', limit=0)
        mdl = core.Model(LungDensityAdapter(), meta, 'ctrate')
        done, nerr, _ = run_chest(a, meta, mdl, torch.device('cpu'), 0, 1)
        assert done == 8 and nerr == 0
        z = np.load(os.path.join(tmp, 'out', 'toy', 'chest_shard0of1.npz'), allow_pickle=True)
        D, V = z['edit_delta'], z['edit_nvox']
        assert D.shape == (8, 4, len(FINDING_NAMES)) and len(json.loads(str(z['lobes']))) == 4
    j = FINDING_NAMES.index('ct_emphysema')
    emph_rows = np.arange(8) % 2 == 0
    assert (V[emph_rows, 0] > 0).all()                       # low-attenuation lung found on emphysema scans
    assert (D[emph_rows, 1, j] < D[emph_rows, 0, j]).all()   # the edit lowers the emphysema score
    lab = pd.DataFrame(rows)[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(float)
    recs = [dict(labels=lab[i].tolist(), delta=D[i], nvox=V[i].tolist()) for i in range(8)]
    assert metrics_edits(recs)['emph']['n_scans'] == int((V[:, 0] > 0).sum())


def test_answer_then_write_helpers():
    """Stage-B 'list' report mode: finding list prefix, prompt and split back into list and report body."""
    from organic.train_stageB import finding_list, report_prompt, split_report
    pre = finding_list(['ct_emphysema', 'ct_pleural_effusion'])
    assert pre.startswith('Abnormal findings: ') and pre.rstrip().endswith('Report:')
    lst, body = split_report(pre + 'The lungs are hyperinflated.')
    assert lst.startswith('Abnormal findings:') and body == 'The lungs are hyperinflated.'
    assert split_report('No acute findings.') == ('', 'No acute findings.')
    assert report_prompt('ctrate', 'list').endswith('then write the report.')
    assert report_prompt('merlin') == 'Write the findings section of the radiology report for this abdominal CT.'


def _run_all():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}', flush=True)
    print(f'SMOKE TEST PASSED ({len(tests)} tests)')


if __name__ == '__main__':
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    _run_all()
