"""Each encoder's own (native) prompt pair for lesion findings that are not in meta.json (2026-10-27), so lesion
detection is scored by exactly the rule of the MLC columns: Model.deltas = sim(pos) - sim(neg) with the adapter's
prompts() (Eq. 1; monotone in every model's own two-prompt softmax, = ItemizedCLIP's positive-only score, = RADAR+'s
ensemble rule). The findings are added to a copy of meta with a label (CT-RATE / Merlin label style, used by CT-CLIP,
ItemizedCLIP, fVLM) and a phrase (Merlin phrase style, used by ours, COLIPRI, Merlin, HLIP, Jolia, SPECTRE, RADAR+).
RADAR+ has no native prompt for any of them (its Merlin list has renal cyst only): its default template is scored on the
named organ's feature, as for its other non-native findings. Jolia scores them on its liver / pancreas / kidneys organ
head, as its own Merlin findings (hepatomegaly -> liver, hydronephrosis -> kidneys)."""
import copy

import torch

LESION_FINDINGS = {   # key: (label, phrase, RADAR+ organ row, Jolia organ head)
    'mr_liver_mass': ('Liver mass', 'liver mass', 'liver', 'liver'),
    'mr_pancreatic_mass': ('Pancreas mass', 'pancreatic mass', 'pancreas', 'pancreas'),
    'mr_renal_mass': ('Kidney mass', 'renal mass', 'kidney', 'kidneys'),
    'mr_left_renal_mass': ('Left kidney mass', 'left renal mass', 'kidney', 'kidneys'),
    'mr_right_renal_mass': ('Right kidney mass', 'right renal mass', 'kidney', 'kidneys'),
}


def native_prompts(ad, meta, keys):
    """-> (tp, tn) encoded positive / negative prompts of the adapter for the findings `keys` (and prints them)."""
    m = copy.deepcopy(meta)
    for f in keys:
        m['label_name'][f], m['phrase'][f] = LESION_FINDINGS[f][:2]
    if type(ad).__module__.endswith('.radar'):
        from extaudit.models import radar
        radar.DEFAULT_ORGAN.update({f: LESION_FINDINGS[f][2] for f in keys if LESION_FINDINGS[f][2]})
    if type(ad).__module__.endswith('.jolia'):         # Jolia scores a finding on its organ head (as hepatomegaly -> liver)
        from extaudit.models import jolia
        jolia.ORGAN.update({f: LESION_FINDINGS[f][3] for f in keys})
    pos, neg = zip(*[ad.prompts(f, m) for f in keys])
    show = lambda s: s.replace('\x00', '').replace('\x01', ' | ')[:120]  # noqa: E731
    print('native prompts:', [(show(p), show(n)) for p, n in zip(pos, neg)], flush=True)
    with torch.no_grad():
        return ad.encode_text(list(pos)), ad.encode_text(list(neg))
