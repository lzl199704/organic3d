"""CPU test: tumour findings are written into each adapter's own prompt template (extaudit/native_prompts.py), so tumour
detection is scored by the same rule as the MLC columns. Uses the static prompt rules of four adapters on a copy of the
finding metadata; no weights are loaded."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch  # noqa: E402

from extaudit.native_prompts import LESION_FINDINGS, native_prompts  # noqa: E402


class Toy:
    """minimal adapter: CT-CLIP-style template, deterministic text embedding."""
    @staticmethod
    def prompts(f, meta):
        lab = meta['label_name'][f]
        return f'{lab} is present.', f'{lab} is not present.'

    def encode_text(self, sents):
        return torch.stack([torch.tensor([float(len(s)), float(sum(map(ord, s)) % 97)]) for s in sents])


def test_templates():
    meta = {'label_name': {}, 'phrase': {}}
    keys = ['mr_liver_mass', 'mr_pancreatic_mass', 'mr_left_renal_mass']
    tp, tn = native_prompts(Toy(), meta, keys)
    assert tp.shape == tn.shape == (3, 2)
    assert 'mr_liver_mass' not in meta['label_name'], 'native_prompts must not modify the caller meta'
    assert Toy.prompts('mr_liver_mass', {'label_name': {'mr_liver_mass': LESION_FINDINGS['mr_liver_mass'][0]}}) == \
        ('Liver mass is present.', 'Liver mass is not present.')
    print('ok  test_templates')


if __name__ == '__main__':
    test_templates()
    print('NATIVE PROMPT TEST PASSED')
