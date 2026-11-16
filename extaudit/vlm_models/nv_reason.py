"""NV-Reason-CT (nvidia/NV-Reason-CT; Qwen3.5-4B + Primus/COLIPRI ViT, 13,824 visual tokens) as an extaudit VLM adapter
(vlm_core.py API), wrapping organic/audit_nvreason.py: the model's own preprocessing
(LPS, 2 mm, its anatomy-aware 192^3 crop for the region) applied to our 1 mm RAS array (affine = identity, as the
audit's .bin path), its question template, and logit(Yes) - logit(No) at the first answer token with the image prefix
encoded once per condition. Run in the organic env (transformers 5.x).
"""
import copy

import numpy as np
import torch

from organic.audit_nvreason import REGION, REPORT, NVReason, crop_slices


class Adapter:
    def __init__(self, ckpt, device):
        self.nv = NVReason(device)
        self.region = 'abdomen'

    def set_dataset(self, dataset):
        self.region = REGION.get(dataset, 'abdomen')

    @staticmethod
    def question(phrase):
        return f'Is {phrase} present in this CT? Answer only Yes or No.'   # = audit_nvreason.nv_question

    @torch.no_grad()
    def encode(self, hu):
        from monai.data import MetaTensor
        nv = self.nv
        arr = hu.float().cpu()[None]
        aff = torch.as_tensor(np.eye(4))
        d = nv.tf_arrays({'image': MetaTensor(arr, affine=aff), 'seg': MetaTensor(torch.zeros_like(arr), affine=aff)})
        img = d['image'].as_subclass(torch.Tensor)
        crop = nv.Crop(keys=['image'], roi_size=nv.il.spatial_size, pixdim=nv.il.pixdim)
        img = img[crop_slices(crop, img, self.region)].permute(0, 3, 2, 1).contiguous()
        self._pv = nv.normalise(img)
        cache, feat = nv.prefix(self._pv)
        return cache, feat

    # free-text report (extaudit/vlm_report.py): the audit's report request (audit_nvreason.REPORT) through NVReason.generate,
    # greedy, as the stage-B evaluator's --gen path (which capped at 512 new tokens; here the driver's cap)
    def report_prompt(self, dataset):
        return REPORT['merlin' if dataset == 'merlin' else 'ctrate']

    @torch.no_grad()
    def report(self, hu, dataset, max_new_tokens=768):
        self.encode(hu)
        return self.nv.generate(self._pv, self.report_prompt(dataset), max_new_tokens=max_new_tokens).strip()

    @torch.no_grad()
    def yes_no_logit(self, cache, questions):
        nv, out = self.nv, []
        for q in questions:
            full = nv.prompt(q)
            ids = nv._ids(full[full.index('<|vision_end|>') + len('<|vision_end|>'):])
            lg = nv.model(input_ids=ids, past_key_values=copy.deepcopy(cache), use_cache=True).logits[0, -1].float()
            out.append(lg[nv.yes] - lg[nv.no])
        return torch.stack(out)

    def p_yes(self, cache, questions):
        return torch.sigmoid(self.yes_no_logit(cache, questions))
