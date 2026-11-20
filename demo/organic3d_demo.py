"""ORGANIC-3D on one CT: the paper's alignment tests for one sentence, with a figure.

For a sentence such as "There is a 9.8 cm mass in the left kidney." the encoder scores S(V, s) on
  clean      the original scan;
  target     the scan with the organs the sentence names erased (TotalSegmentator masks dilated 2 mm, filled with fat,
             -100 HU, plus noise of SD 15 HU, as in the benchmark);
  control    the same edit of a control organ the sentence does not name (default: the first organ group of the
             benchmark's list that is neither named nor related);
  side swap  the sentence with left and right swapped, on the original scan (sentences that name a side).
An aligned model lowers the score more after the target edit than after the control edit, and prefers the sentence
that names the correct side.

    python demo/organic3d_demo.py --ct ct.nii.gz --seg seg.nii.gz --region abdomen \
        --sentence "There is a 9.8 cm mass in the left kidney." --out example.png [--gif example.gif]
    python demo/organic3d_demo.py --app                    # interactive web interface (needs gradio)

--seg: TotalSegmentator v2 'total' multilabel mask on the CT's 1 mm RAS grid; without it, TotalSegmentator is run
(scripts/inference/organic3d.py). --weights: the encoder repo or a local directory.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts', 'inference'))
import organic3d as O                                                                # noqa: E402
from organic.encoder.evaluate import EVAL_ORGANS                                      # noqa: E402
from organic.encoder.routing import sentence_organs                                          # noqa: E402
from organic.vocab import ORGAN_NAMES, ORGANS, RELATED                               # noqa: E402

FILL_HU, NOISE_SD, DILATE_MM = -100.0, 15.0, 2


def swap_side(s):
    t = re.sub(r'\bleft\b', '\0', s, flags=re.I)
    t = re.sub(r'\bright\b', 'left', t, flags=re.I)
    return t.replace('\0', 'right')


def control_for(targets, region):
    """First organ group of the benchmark's erasure list that is not named by the sentence and not related to it."""
    bad = set(targets) | {r for t in targets for r in RELATED.get(t, ())}
    for g in EVAL_ORGANS[O.DATASET[region]]:
        g = [g] if isinstance(g, str) else g
        if not bad & set(g):
            return g
    raise ValueError('no control organ available')


def organ_mask(seg, groups):
    from scipy import ndimage
    m = np.isin(seg, [i for g in groups for i in ORGANS[g]])
    return ndimage.binary_dilation(m, iterations=DILATE_MM) if m.any() else m


def erase(hu, mask, seed=0):
    out = hu.copy()
    rng = np.random.default_rng(seed)
    out[mask] = FILL_HU + rng.normal(0, NOISE_SD, int(mask.sum())).astype(np.float32)
    return out


def run(enc, hu, seg, region, sentence, control=None, device='cuda'):
    """-> dict of scores and the volumes for the figure."""
    targets = [ORGAN_NAMES[i] for i in sentence_organs(sentence)]
    if not targets:
        raise ValueError('the sentence names no organ, so there is nothing to erase')
    control = control or control_for(targets, region)
    centre = O.crop_centre(hu, region, seg)
    mt, mc = organ_mask(seg, targets), organ_mask(seg, control)
    vols = {'clean': hu, 'target': erase(hu, mt, 1), 'control': erase(hu, mc, 2)}
    S = {k: float(enc.similarity(enc.encode_image(O.model_input(v, centre, device)), [sentence])[0])
         for k, v in vols.items()}
    res = {'sentence': sentence, 'target_organs': targets, 'control_organs': control,
           'score_clean': S['clean'], 'score_target_erased': S['target'], 'score_control_erased': S['control'],
           'drop_target': S['clean'] - S['target'], 'drop_control': S['clean'] - S['control']}
    res['organ_dependence'] = res['drop_target'] > res['drop_control']
    sw = swap_side(sentence)
    if sw != sentence:
        T = enc.encode_image(O.model_input(hu, centre, device))
        s_true, s_swap = enc.similarity(T, [sentence, sw]).tolist()
        res.update(side_swapped=sw, score_side_swapped=s_swap, correct_side=s_true > s_swap)
    return res, vols, mt, mc


TARGET_C, CONTROL_C, BASE_C, SIDE_C = '#f2c200', '#2fa36b', '#5b6b7f', '#8a79b0'   # as in the paper's Fig. 1
WIN = (-160, 240)


def _nice(groups):
    return ' + '.join(' '.join(g.split('_')[::-1]) if g.split('_')[-1] in ('left', 'right') else g.replace('_', ' ')
                      for g in groups)


def _views(res, vols, mt, mc):
    """Coronal plane through the target organ, cropped to both edited organs; radiological view (patient right on
    image left). -> function volume -> 2-D image."""
    both = mt | mc
    y = int(np.round(np.argwhere(mt)[:, 1].mean()))
    xs, zs = np.where(both[:, y, :])
    pad = 40
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, both.shape[0])
    z0, z1 = max(zs.min() - pad, 0), min(zs.max() + pad, both.shape[2])
    return lambda v: v[x0:x1, y, z0:z1].T[::-1, ::-1]


def _bars(ax, res, show):
    """Horizontal score bars for the conditions in `show` (subset of clean, target, control, side)."""
    rows = [('clean', 'Original', res['score_clean'], BASE_C),
            ('target', f"Target erased ({_nice(res['target_organs'])})", res['score_target_erased'], TARGET_C),
            ('control', f"Control erased ({_nice(res['control_organs'])})", res['score_control_erased'], CONTROL_C)]
    if 'score_side_swapped' in res:
        rows.append(('side', 'Wrong side named', res['score_side_swapped'], SIDE_C))
    vals = [r[2] for r in rows]
    lo, hi = min(0.0, min(vals)), max(vals)
    span = hi - lo
    ax.set_xlim(lo - 0.04 * span, hi + 0.42 * span)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.axvline(0, color='#c9ced6', lw=0.8)
    ax.set_yticks(range(len(rows)))                                    # labels right-aligned, left of the axis
    ax.set_yticklabels([r[1] for r in rows], fontsize=9, color='#2b3440')
    ax.tick_params(axis='y', length=0, pad=6)
    for k, (key, label, v, c) in enumerate(rows):
        if key not in show:
            continue
        ax.barh(k, v, color=c, height=0.55)
        txt = f'{v:.2f}'
        if key in ('target', 'control', 'side'):
            txt += f'   ({_signed(v - res["score_clean"])})'
        ax.text(max(v, 0) + 0.02 * span, k, txt.replace('-', '\u2212'), va='center', fontsize=9, color='#2b3440')
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('Score of the sentence, S(V, s)', fontsize=9, color='#5b6470', loc='left')


def _signed(x):
    return f'{x:+.2f}'.replace('-', '\u2212')


def _verdict(res):
    tick = lambda b: '\u2713' if b else '\u2717'                                    # noqa: E731
    msg = f"{tick(res['organ_dependence'])} organ dependence: score change {_signed(-res['drop_target'])} after " \
          f"erasing the target vs {_signed(-res['drop_control'])} after the control"
    if 'correct_side' in res:
        msg += f"     {tick(res['correct_side'])} correct side preferred"
    return msg


def _sides(ax):
    for x, t in ((0.02, 'R'), (0.98, 'L')):                            # radiological view
        ax.text(x, 0.03, t, transform=ax.transAxes, ha='left' if t == 'R' else 'right', va='bottom', color='white',
                fontsize=9)


def figure(res, vols, mt, mc, path=None):
    """Static summary: original, target-erased and control-erased slices, and the score of each condition."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    view = _views(res, vols, mt, mc)
    fig = plt.figure(figsize=(14, 3.9), facecolor='white')
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.75, 1.25], wspace=0.05)
    panels = (('clean', None, 'Original', None), ('target', mt, 'Erase target', _nice(res['target_organs'])),
              ('control', mc, 'Erase control', _nice(res['control_organs'])))
    for k, (key, m, title, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        ax.imshow(view(vols[key]), cmap='gray', vmin=WIN[0], vmax=WIN[1])
        if m is not None:
            c, ls = (TARGET_C, 'solid') if key == 'target' else (CONTROL_C, 'dashed')
            ax.contour(view(m).astype(float), levels=[0.5], colors=[c], linewidths=1.6, linestyles=ls)
        ax.set_title(title + (f'\n{sub}' if sub else '\n '), fontsize=10)
        ax.axis('off')
        _sides(ax)
    _bars(fig.add_subplot(gs[4]), res, {'clean', 'target', 'control', 'side'})
    fig.suptitle(f'"{res["sentence"]}"', fontsize=11, y=1.02)
    fig.text(0.5, -0.04, _verdict(res), ha='center', fontsize=10, color='#2b3440')
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def animation(res, vols, mt, mc, path, dpi=100):
    """Short GIF of the tests: erase the target organ, restore, erase a control organ, swap the side, verdict.
    Only the four computed conditions carry scores; the frames between them are visual transitions."""
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    view = _views(res, vols, mt, mc)
    clean, tgt, ctl = view(vols['clean']), view(vols['target']), view(vols['control'])
    mtv, mcv = view(mt).astype(float), view(mc).astype(float)
    sent, swapped = res['sentence'], res.get('side_swapped')
    # (image, contour, caption, bars shown, sentence shown, duration ms)
    seq = [(clean, None, '1  Score the sentence on the original scan', {'clean'}, sent, 1600)]
    for a in (0.35, 0.7):
        seq.append(((1 - a) * clean + a * tgt, ('t', mtv), '2  Erase the organ the sentence names', {'clean'}, sent, 220))
    seq.append((tgt, ('t', mtv), '2  Erase the organ the sentence names: the score should drop', {'clean', 'target'},
                sent, 2200))
    for a in (0.35, 0.7):
        seq.append(((1 - a) * clean + a * ctl, ('c', mcv), '3  Restore it, erase a control organ instead',
                    {'clean', 'target'}, sent, 220))
    seq.append((ctl, ('c', mcv), '3  Erase a control organ: the score should stay', {'clean', 'target', 'control'},
                sent, 2200))
    if swapped:
        seq.append((clean, None, '4  Name the wrong side: the score should be lower',
                    {'clean', 'target', 'control', 'side'}, swapped, 2200))
    seq.append((clean, None, _verdict(res), {'clean', 'target', 'control', 'side'}, sent, 3200))
    frames, durs = [], []
    for img, cont, cap, show, sen, ms in seq:
        fig = plt.figure(figsize=(11.5, 3.7), facecolor='white')
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.6, 1.25], wspace=0.04, left=0.02, right=0.98, top=0.82, bottom=0.12)
        ax = fig.add_subplot(gs[0])
        ax.imshow(img, cmap='gray', vmin=WIN[0], vmax=WIN[1])
        if cont is not None:
            c, ls = (TARGET_C, 'solid') if cont[0] == 't' else (CONTROL_C, 'dashed')
            ax.contour(cont[1], levels=[0.5], colors=[c], linewidths=1.8, linestyles=ls)
        ax.axis('off')
        _sides(ax)
        _bars(fig.add_subplot(gs[2]), res, show)
        col = SIDE_C if sen != sent else '#1f2933'
        fig.text(0.5, 0.93, f'"{sen}"', ha='center', fontsize=12, color=col)
        fig.text(0.5, 0.03, cap, ha='center', fontsize=10.5, color='#2b3440')
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, facecolor='white')
        plt.close(fig)
        frames.append(Image.open(buf).convert('RGB'))
        durs.append(ms)
    pal = frames[-2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True)
    return path


def load(ct, seg, is_1mm):
    import nibabel as nib
    hu, aff = O.load_ct(ct, resample=not is_1mm)
    if seg:
        m = np.asarray(nib.as_closest_canonical(nib.load(seg)).dataobj).astype(np.uint8)
        assert m.shape == hu.shape, f'mask {m.shape} is not on the 1 mm grid {hu.shape}'
    else:
        m = O.totalsegmentator(hu, aff)
    return hu, m


def app(weights, device):
    import gradio as gr
    encs = {}

    def go(ct, seg, region, sentence, is_1mm):
        if region not in encs:
            encs[region] = O.Encoder(weights, device, region=region)
        hu, m = load(ct, seg, is_1mm)
        res, vols, mt, mc = run(encs[region], hu, m, region, sentence, device=device)
        return figure(res, vols, mt, mc), json.dumps(res, indent=1)
    with gr.Blocks(title='ORGANIC-3D alignment tests') as ui:
        gr.Markdown('# ORGANIC-3D: does the score follow the image?\nUpload a CT (NIfTI, HU) and optionally its '
                    'TotalSegmentator v2 mask, then write a sentence that names an organ.')
        with gr.Row():
            ct = gr.File(label='CT (NIfTI)', type='filepath')
            seg = gr.File(label='TotalSegmentator mask (optional)', type='filepath')
        with gr.Row():
            region = gr.Radio(['chest', 'abdomen'], value='abdomen', label='Region')
            is_1mm = gr.Checkbox(False, label='CT is already 1 mm RAS')
        sentence = gr.Textbox('There is a mass in the left kidney.', label='Sentence')
        btn = gr.Button('Run the tests')
        plot, out = gr.Plot(), gr.Code(language='json', label='Scores')
        btn.click(go, [ct, seg, region, sentence, is_1mm], [plot, out])
    ui.launch()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--app', action='store_true', help='start the interactive web interface')
    ap.add_argument('--ct')
    ap.add_argument('--seg', default=None)
    ap.add_argument('--is_1mm', action='store_true')
    ap.add_argument('--region', choices=['chest', 'abdomen'], default='abdomen')
    ap.add_argument('--sentence')
    ap.add_argument('--control', default=None, help='control organ groups, comma-separated (default: automatic)')
    ap.add_argument('--weights', default=O.ENCODER_REPO)
    ap.add_argument('--out', default='organic3d_example.png')
    ap.add_argument('--gif', default=None, help='also write a short animation of the tests (.gif)')
    a = ap.parse_args()
    dev = 'cuda'
    if a.app:
        return app(a.weights, dev)
    enc = O.Encoder(a.weights, dev, region=a.region)
    hu, m = load(a.ct, a.seg, a.is_1mm)
    res, vols, mt, mc = run(enc, hu, m, a.region, a.sentence, a.control.split(',') if a.control else None, dev)
    figure(res, vols, mt, mc, a.out)
    if a.gif:
        animation(res, vols, mt, mc, a.gif)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.exit(main())
