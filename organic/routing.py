"""Fixed (rule-based) sentence-to-slot routing for the `model.pool: fixed` ablation (2026-10-16).

The final model (`pool: sentence`) lets each sentence learn which anatomy slots to pool (beta over all 144 slots).
The ablation replaces that choice by a rule, as in concept-query models (Jolia / ConQuer pairs query k with its own
report section): a sentence may pool only the slots of the organ groups it names; the weights among those slots stay
learned. A sentence that names no organ group pools the 16 global slots.

Organs are read from the sentence text:
  * a canonical prompt ("There is {phrase}." / "No {phrase}.") takes its finding's organ groups from vocab.FINDINGS
    (organ-less findings -> global slots, as in the vocabulary);
  * any other sentence is matched against the keyword lists below; paired organs (lungs, kidneys) take the side named
    in the sentence (left only / right only), otherwise both sides.
"""
import re

import torch

from .vocab import FINDINGS, ORGAN_ID

_CANON = {}
for _f, (_col, _orgs, _p) in FINDINGS.items():
    for _s in (f'There is {_p}.', f'No {_p}.'):
        _CANON[_s.lower()] = [ORGAN_ID[o] for o in _orgs]

_KW = [  # organ group (or 'lungs' / 'kidneys' = side decided by the sentence) -> regex on the lower-cased sentence
    ('lungs', r'lung|pulmonar|lobe|lobar|lingul|bronch|alveol|interstitial|emphysem|atelecta|consolidat|ground[- ]glass|'
              r'mosaic|septal|honeycomb|fibrotic|reticul|air[- ]?trapping|tree[- ]in[- ]bud'),
    ('heart', r'heart|cardi|pericardi|coronar|atri(um|al)|ventric'),
    ('aorta', r'aort|arterial wall'),
    ('esophagus', r'esophag|oesophag|hiatal|hiatus'),
    ('trachea', r'trache'),
    ('liver', r'liver|hepat|biliary|bile duct|steatosis'),
    ('gallbladder', r'gallbladder|gall bladder|gallstone|cholecyst|cholelith'),
    ('spleen', r'splen'),
    ('pancreas', r'pancrea'),
    ('kidneys', r'kidney|renal|hydronephr|nephrolith'),
    ('stomach', r'stomach|gastric|hiatal|hiatus'),
    ('small_bowel', r'small bowel|small intestin|duoden|jejun|ileum|ileal|bowel obstruction'),
    ('colon', r'colon|colonic|diverticul|append|cecum|caecum|sigmoid|rect(um|al)'),
    ('pelvic_organs', r'(?<!gall )(?<!gall)bladder|prostat'),
]
_KW = [(o, re.compile(p)) for o, p in _KW]
_LEFT, _RIGHT = re.compile(r'\bleft\b'), re.compile(r'\bright\b')


def sentence_organs(text):
    """-> list of organ-group ids named by the sentence ([] = none)."""
    t = text.lower().strip()
    if t in _CANON:
        return _CANON[t]
    left, right = bool(_LEFT.search(t)), bool(_RIGHT.search(t))
    out = []
    for o, rx in _KW:
        if not rx.search(t):
            continue
        if o in ('lungs', 'kidneys'):
            base = 'lung' if o == 'lungs' else 'kidney'
            sides = ['left'] if (left and not right) else ['right'] if (right and not left) else ['left', 'right']
            out += [ORGAN_ID[f'{base}_{s}'] for s in sides]
        else:
            out.append(ORGAN_ID[o])
    return sorted(set(out))


def route_matrix(strings, organ_of_query, device=None):
    """(S, K) float mask: 1 where sentence s may pool slot k (its organs' slots, or the global slots if none)."""
    oq = organ_of_query.cpu()
    R = torch.zeros(len(strings), len(oq))
    for i, s in enumerate(strings):
        ids = sentence_organs(s)
        R[i] = torch.isin(oq, torch.tensor(ids)).float() if ids else (oq < 0).float()
        if R[i].sum() == 0:                       # organ named but no slots for it (cannot happen with anatomy queries)
            R[i] = (oq < 0).float()
    return R.to(device) if device is not None else R
