"""Jolia's text side for the ReXGroundingCT finding sentences, pre-encoded once with Qwen3-Embedding-8B (last-token
pooled, as the Jolia adapter's text cache) so the grounding shards need not load the 16 GB text encoder. A separate
file: the shared Jolia cache is not touched; extaudit/core.py run_grounding merges it into the Jolia adapter's cache.

    CUDA_VISIBLE_DEVICES=0 $ORGANIC3D_BASELINES/jolia/venv/bin/python scripts/data/rex_jolia_text_cache.py
Jolia's bundled code is read from $ORGANIC3D_BASELINES/jolia/hf (the HF raidium/Jolia snapshot, extaudit/README.md);
{work_root} from organic/workroot.py ($ORGANIC3D_WORK_ROOT, else paths.work_root of $ORGANIC3D_ENV).
Writes {work_root}/rexground/jolia_text_cache.pt ({text: float32 (4096,)}).
"""
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extaudit.paths import baseline                         # noqa: E402
sys.path.insert(0, baseline('jolia', 'hf'))
from extaudit.models.jolia import QWEN, QWEN_REV, encode_pooled   # noqa: E402
from text_encoder_jolia import JoliaTextEncoder            # noqa: E402

from organic.workroot import work_root                              # noqa: E402

W = work_root()


def main():
    t = pd.read_parquet(os.path.join(W, 'rexground', 'findings.parquet'))
    texts = sorted(set(t.text))
    out = os.path.join(W, 'rexground', 'jolia_text_cache.pt')
    old = torch.load(out) if os.path.exists(out) else {}
    todo = [x for x in texts if x not in old]
    print(len(texts), 'texts,', len(todo), 'to encode', flush=True)
    if not todo:
        return
    from huggingface_hub import snapshot_download
    qwen = snapshot_download(QWEN, revision=QWEN_REV)                    # the exact snapshot used in the paper
    te = JoliaTextEncoder.from_pretrained(qwen, dtype=torch.bfloat16, context_length=512).to('cuda').eval()
    feat = dict(old)
    with torch.no_grad():
        for i in range(0, len(todo), 64):
            feat.update(zip(todo[i:i + 64], encode_pooled(te, todo[i:i + 64]).cpu()))
    torch.save(feat, out)
    print('wrote', out, len(feat), flush=True)


if __name__ == '__main__':
    main()
