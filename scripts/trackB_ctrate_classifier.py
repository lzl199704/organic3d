"""CT-RATE's official text classifier (the RadBERT multi-abnormality classifier the CT-RATE authors used to produce
dataset/multi_abnormality_labels), as the single CT-RATE extractor of report-level Track B.

Weights: HF dataset ibrahimhamamci/CT-RATE, models/RadBertClassifier.pth (CC BY-NC-SA 4.0, downloaded 2026-10-14), on
the base model zzxslp/RadBERT-RoBERTa-4m; both under {CLS_DIR} (default {work_root}/evalTrackB/ctrate_classifier,
override with $CTRATE_CLASSIFIER_DIR; layout models/RadBertClassifier.pth and RadBERT-RoBERTa-4m/). Code = CT-CLIP's
text_classifier (github.com/ibrahimethemhamamci/CT-CLIP, classifier.py, dataset.py, model_trainer.infer): RobertaModel pooler output -> Linear(768,
18), tokenizer max_length 512 (padding to max_length, truncation), sigmoid >= 0.5. Label order = infer.py label_cols
(= the 18 label_ct_* findings of organic.vocab, same order). Reference text = Findings_EN + ' ' + Impressions_EN
(the classifier's training text, text_classifier/data/train.csv).

    python scripts/trackB_ctrate_classifier.py      # check: reference reports vs released labels
Locations: {work_root} and {ctrate_root} from scripts/workroot.py ($ORGANIC3D_ENV, default configs/env/local.yaml).
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workroot import env_paths, work_root  # noqa: E402

W = work_root()
CTR = str(env_paths().ctrate_root)
CLS_DIR = os.environ.get('CTRATE_CLASSIFIER_DIR', f'{W}/evalTrackB/ctrate_classifier')
CKPT = f'{CLS_DIR}/models/RadBertClassifier.pth'
BASE = f'{CLS_DIR}/RadBERT-RoBERTa-4m'
CACHE = f'{W}/evalTrackB/ctrate_classifier_cache.parquet'
REPORTS = f'{CTR}/dataset/radiology_text_reports/validation_reports.csv'
RELEASED = f'{CTR}/dataset/multi_abnormality_labels/valid_predicted_labels.csv'
LABEL_COLS = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
              'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
              'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
              'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
              'Interlobular septal thickening']


class RadBertClassifier(nn.Module):
    """classifier.py, with the base model read from the local copy."""
    def __init__(self, n_classes=18):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.config = AutoConfig.from_pretrained(BASE)
        self.model = AutoModel.from_pretrained(BASE, config=self.config)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attn_mask):
        return self.classifier(self.model(input_ids=input_ids, attention_mask=attn_mask).pooler_output)


def _key(t):
    return hashlib.sha1(t.encode('utf-8')).hexdigest()


_LOADED = {}


def _model(device=None):
    """Tokenizer and classifier, loaded once per process."""
    if not _LOADED:
        from transformers import AutoTokenizer
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = RadBertClassifier(len(LABEL_COLS))
        st = torch.load(CKPT, map_location='cpu')
        st.pop('model.embeddings.position_ids', None)            # buffer saved by older transformers (= arange)
        m.load_state_dict(st)
        _LOADED.update(tok=AutoTokenizer.from_pretrained(BASE, do_lower_case=True), m=m.to(device).eval(), device=device)
    return _LOADED['tok'], _LOADED['m'], _LOADED['device']


def classify(texts, device=None, bs=32):
    """texts -> (n, 18) sigmoid probabilities (threshold 0.5 = the released labels). Cached by text hash."""
    texts = [t if isinstance(t, str) and t.strip() else ' ' for t in texts]   # dataset.py: NaN -> ' '
    cache = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame(columns=['key'] + LABEL_COLS)
    have = set(cache.key)
    todo = list(dict.fromkeys(t for t in texts if _key(t) not in have))
    if todo:
        tok, m, device = _model(device)
        P = []
        with torch.no_grad():
            for s in range(0, len(todo), bs):
                enc = tok(todo[s:s + bs], return_tensors='pt', max_length=512, padding='max_length', truncation=True)
                P.append(torch.sigmoid(m(enc['input_ids'].to(device), enc['attention_mask'].to(device)).float()).cpu().numpy())
        new = pd.DataFrame(np.concatenate(P), columns=LABEL_COLS)
        new.insert(0, 'key', [_key(t) for t in todo])
        cache = pd.concat([cache, new], ignore_index=True)
        cache.to_parquet(CACHE)
    c = cache.set_index('key')
    return c.loc[[_key(t) for t in texts], LABEL_COLS].to_numpy(np.float64)


def reference_texts(volumes):
    """CT-RATE validation reports of these volumes as the classifier read them: Findings_EN + ' ' + Impressions_EN."""
    r = pd.read_csv(REPORTS).set_index('VolumeName')
    r = r.loc[list(volumes)]
    return (r.Findings_EN.fillna('').astype(str) + ' ' + r.Impressions_EN.fillna('').astype(str)).tolist()


def released_labels(volumes):
    return pd.read_csv(RELEASED).set_index('VolumeName').loc[list(volumes), LABEL_COLS].to_numpy(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200, help='first N scans of the CT-RATE audit sample')
    a = ap.parse_args()
    s = pd.read_parquet(f'{W}/extaudit/ctrate_audit1000_set.parquet').head(a.n)
    P = classify(reference_texts(s.volume)) >= 0.5
    R = released_labels(s.volume)
    agree = (P == R).mean(0)
    print(f'classifier on {len(s)} reference reports vs released labels: agreement {agree.mean():.4f} '
          f'(min {agree.min():.3f}, {LABEL_COLS[int(agree.argmin())]}); scans identical on all 18: {(P == R).all(1).mean():.3f}')


if __name__ == '__main__':
    main()
