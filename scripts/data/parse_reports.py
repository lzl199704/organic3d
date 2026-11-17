"""Parse report sentences into claims with Qwen3-14B (vLLM), for stage-A text targets and Merlin labels.

Each report is split into sentences here (deterministic). Pass 1: schema-constrained JSON per report
(finding = vocab key or 'other', polarity, organ). Side and severity come from rules. Pass 2: for sentences
with a present abnormality, a negated and an organ-swapped rewrite (schema-constrained JSON).

Outputs ({work_root}/parsed/):
  claims.parquet        volume, text, finding, polarity, organ, side, severity, neg_text, swap_text
  merlin_labels.parquet volume, label_<mr_finding>  (1 present; 0 absent or organ reported without it; NaN otherwise)

Usage (one process per GPU; merge afterwards with --merge):
  CUDA_VISIBLE_DEVICES=0 <vllm env>/bin/python scripts/data/parse_reports.py --env configs/env/local.yaml --shard 0/4
  <vllm env>/bin/python scripts/data/parse_reports.py --env configs/env/local.yaml --merge
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
from organic.vocab import FINDINGS, ORGAN_NAMES, MERLIN_FINDINGS
from organic.data.dataset import split_sentences

LLM = 'Qwen/Qwen3-14B'
SPLITS = ['train', 'val']
POL = {'present', 'absent', 'uncertain', 'none'}
SIDES = {'left', 'right', 'bilateral', 'none'}
SEV = {'mild', 'moderate', 'severe', 'none'}

DS_KEYS = {'merlin': [k for k in FINDINGS if k.startswith('mr_')], 'ctrate': [k for k in FINDINGS if k.startswith('ct_')]}
VOCAB = {d: '\n'.join(f'{k}: {FINDINGS[k][2]}' for k in ks) for d, ks in DS_KEYS.items()}
PROMPT = """You label every numbered sentence of a CT radiology report.
For each sentence give:
- finding: the key below ONLY if the sentence explicitly states that exact finding; otherwise "other".
  Cysts, hypodensities, nodular contours, postoperative changes, surgical gas, subcentimeter or non-enlarged
  lymph nodes and anything not in the list are "other". A sentence denying the finding still gets its key.
- polarity: "present" if the finding/abnormality is stated as present, "absent" if stated as absent or the
  structure is normal, "uncertain" if hedged, "none" if the sentence states no finding.
- organ: the organ group the sentence describes, or "other".
Return JSON: {{"items": [{{"i": 0, "finding": ..., "polarity": ..., "organ": ...}}, ...]}} covering all sentences.

Finding keys:
{vocab}

Report sentences:
{sents}"""

REWRITE = """Rewrite this CT report sentence two ways. Return JSON {{"negated": ..., "organ_swap": ...}}.
negated: minimal edit stating the abnormality is ABSENT; keep all other words.
organ_swap: keep the structure and severity words but move the abnormality to the {target}.
Sentence: {s}"""

SIDE_RE = {'bilateral': re.compile(r'\b(bilateral|bilaterally|both)\b', re.I),
           'left': re.compile(r'\bleft\b', re.I), 'right': re.compile(r'\bright\b', re.I)}
SEV_RE = {'severe': re.compile(r'\b(severe|severely|marked|markedly|massive|large)\b', re.I),
          'moderate': re.compile(r'\b(moderate|moderately)\b', re.I),
          'mild': re.compile(r'\b(mild|mildly|minimal|minimally|slight|small|trace|tiny)\b', re.I)}
SWAP_TARGETS = ['liver', 'gallbladder', 'spleen', 'pancreas', 'left kidney', 'right kidney', 'colon', 'small bowel',
                'aorta', 'stomach', 'bladder', 'left lung', 'right lung', 'heart', 'esophagus']


def side_of(t):
    if SIDE_RE['bilateral'].search(t) or (SIDE_RE['left'].search(t) and SIDE_RE['right'].search(t)):
        return 'bilateral'
    return 'left' if SIDE_RE['left'].search(t) else 'right' if SIDE_RE['right'].search(t) else 'none'


_ANAT_SIZE = re.compile(r'\b(small|large)\s+(bowel|intestin\w*|saphenous|airways?|vessels?|cell)\b', re.I)


def severity_of(t):
    t = _ANAT_SIZE.sub(' ', t)
    for k in ('severe', 'moderate', 'mild'):
        if SEV_RE[k].search(t):
            return k
    return 'none'


def schema(n, dataset):
    item = {'type': 'object', 'properties': {
        'i': {'type': 'integer'}, 'finding': {'enum': DS_KEYS[dataset] + ['other']},
        'polarity': {'enum': sorted(POL)}, 'organ': {'enum': ORGAN_NAMES + ['other']}},
        'required': ['i', 'finding', 'polarity', 'organ']}
    return {'type': 'object', 'properties': {'items': {'type': 'array', 'items': item, 'minItems': n, 'maxItems': n}},
            'required': ['items']}


REWRITE_SCHEMA = {'type': 'object', 'properties': {'negated': {'type': 'string'}, 'organ_swap': {'type': 'string'}},
                  'required': ['negated', 'organ_swap']}


def load_env(p):
    return OmegaConf.load(p)


def build_jobs(cfg):
    rows = []
    for d in ('merlin', 'ctrate'):
        m = pd.read_parquet(os.path.join(cfg.paths.work_root, 'manifests', f'{d}.parquet'))
        m = m[m.split.isin(SPLITS)]
        rows.append(m[['dataset', 'volume', 'report']])
    df = pd.concat(rows, ignore_index=True)
    df['report'] = df.report.fillna('').astype(str)
    uniq = df.drop_duplicates(['dataset', 'report']).reset_index(drop=True)
    uniq['rid'] = np.arange(len(uniq))
    df = df.merge(uniq[['dataset', 'report', 'rid']], on=['dataset', 'report'])
    return df, uniq


def run_shard(cfg, shard, bs):
    import json
    from vllm import LLM as VLLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    df, uniq = build_jobs(cfg)
    k, n = map(int, shard.split('/'))
    uniq = uniq.iloc[k::n].reset_index(drop=True)
    out_dir = os.path.join(cfg.paths.work_root, 'parsed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/shard{k}of{n}.parquet'
    done = set(pd.read_parquet(out_path).rid) if os.path.exists(out_path) else set()
    todo = uniq[~uniq.rid.isin(done)]
    print(f'shard {shard}: {len(uniq)} unique reports, {len(todo)} to do', flush=True)
    llm = VLLM(model=LLM, dtype='bfloat16', max_model_len=8192, gpu_memory_utilization=0.9, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    organs = ', '.join(ORGAN_NAMES)

    def chat(c):
        return tok.apply_chat_template([{'role': 'user', 'content': c}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    rows = [pd.read_parquet(out_path)] if done else []
    rng = np.random.default_rng(k)
    for s0 in range(0, len(todo), bs):
        chunk = todo.iloc[s0:s0 + bs]
        sents = [split_sentences(r) for r in chunk.report]
        keep = [i for i, ss in enumerate(sents) if ss]
        dsets = list(chunk.dataset)
        prompts = [chat(PROMPT.format(vocab=VOCAB[dsets[j]], sents='\n'.join(f'{i} {t}' for i, t in enumerate(sents[j]))))
                   for j in keep]
        sps = [SamplingParams(temperature=0.0, max_tokens=4096,
                              guided_decoding=GuidedDecodingParams(json=schema(len(sents[j]), dsets[j]))) for j in keep]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        recs = []
        for j, o in zip(keep, outs):
            try:
                items = {int(x['i']): x for x in json.loads(o.outputs[0].text)['items']}
            except Exception:
                items = {}
            for i, t in enumerate(sents[j]):
                x = items.get(i, {})
                recs.append(dict(rid=int(chunk.rid.iloc[j]), idx=i, text=t, parsed=i in items,
                                 finding=x.get('finding', 'other'), polarity=x.get('polarity', 'none'),
                                 organ=x.get('organ', 'other'), side=side_of(t), severity=severity_of(t),
                                 neg_text=None, swap_text=None))
        # rewrites for present-abnormality sentences
        pres = [r for r in recs if r['polarity'] == 'present']
        if pres:
            rp = [chat(REWRITE.format(s=r['text'], target=SWAP_TARGETS[int(rng.integers(len(SWAP_TARGETS)))])) for r in pres]
            ro = llm.generate(rp, SamplingParams(temperature=0.0, max_tokens=256,
                                                 guided_decoding=GuidedDecodingParams(json=REWRITE_SCHEMA)), use_tqdm=False)
            for r, o in zip(pres, ro):
                try:
                    d = json.loads(o.outputs[0].text)
                    r['neg_text'] = d.get('negated') or None
                    r['swap_text'] = d.get('organ_swap') or None
                except Exception:
                    pass
        rows.append(pd.DataFrame(recs))
        pd.concat(rows, ignore_index=True).to_parquet(out_path, index=False)
        print(f'{min(s0 + bs, len(todo))}/{len(todo)}', flush=True)


# a vocabulary finding is kept only if the sentence contains one of its own terms (post-hoc gate on LLM labels)
GATES = {
    'mr_hydronephrosis': r'hydroneph|hydroureter|pelviectasis|pelvocaliectasis|caliectasis',
    'mr_renal_calculus': r'calcul|stone|nephrolith|urolith',
    'mr_gallstones': r'gallstone|cholelith|stones?\b.*gallbladder|gallbladder.*\bstones?',
    'mr_gb_distension': r'distend|distension|distention|hydrops',
    'mr_biliary_dilatation': r'biliary|bile duct|\bcbd\b|choledoch|ductal dilat',
    'mr_hepatomegaly': r'hepatomegaly|liver (is )?(enlarged|large)|enlarged liver',
    'mr_fatty_liver': r'steatosis|fatty|hypoattenuat|low[- ]attenuation liver',
    'mr_hiatal_hernia': r'hiatal|hiatus',
    'mr_pancreatic_atrophy': r'atroph',
    'mr_pancreatic_duct': r'pancreatic duct|duct of wirsung|\bpd\b|ductal dilat',
    'mr_splenomegaly': r'splenomegaly|spleen (is )?(enlarged|large)|enlarged spleen',
    'mr_appendicitis': r'appendicitis',
    'mr_diverticulosis': r'diverticulosis|(colon|colonic|sigmoid|descending|ascending|cecal|cecum).{0,40}diverticul|diverticul.{0,40}(colon|sigmoid)',
    'mr_bowel_obstruction': r'obstruct|transition point|\bsbo\b',
    'mr_aortic_aneurysm': r'aneurysm|ectasia|ectatic',
    'mr_aortic_stent': r'stent|endograft|evar',
    'mr_prostatomegaly': r'prostat',
    'mr_ascites': r'ascites|fluid',
    'mr_lymphadenopathy': r'lymphadenopath|adenopathy|enlarged.{0,60}nodes?|nodes?.{0,60}enlarged|nodes?.{0,40}\d(\.\d)? ?(x \d(\.\d)? ?)?cm',
    'mr_uterine_fibroid': r'fibroid|leiomyom',
    'mr_ovarian_cyst': r'(?=.*(ovar|adnex))(?=.*cyst)',
    'ct_medical_material': r'catheter|tube|stent|pacemaker|device|port\b|clip|prosthe|wire|electrode|drain',
    'ct_arterial_calcification': r'atheroscl|atherom|(aort|arter|vascul|valv|coronary|carotid|brachiocephalic|subclavian).{0,60}calcif|calcif.{0,60}(aort|arter|vascul|valv|coronary)',
    'ct_cardiomegaly': r'cardiomegaly|heart (is )?(enlarged|large)|cardiac enlargement',
    'ct_pericardial_effusion': r'pericard',
    'ct_coronary_calcification': r'coronary',
    'ct_hiatal_hernia': r'hiatal|hiatus',
    'ct_lymphadenopathy': r'lymphadenopath|lymph node|adenopathy',
    'ct_emphysema': r'emphysem',
    'ct_atelectasis': r'atelecta',
    'ct_lung_nodule': r'^(?!.*\b(thyroid|liver|hepatic|kidney|renal|adrenal|breast|spleen|splenic|pancrea|bone|vertebra)).*nodul',
    'ct_lung_opacity': r'opacit|ground[- ]glass|density|densities',
    'ct_fibrotic_sequela': r'fibro|sequela|scar',
    'ct_pleural_effusion': r'pleural effusion|effusion',
    'ct_mosaic_attenuation': r'mosaic',
    'ct_peribronchial_thickening': r'peribronch|bronchial wall',
    'ct_consolidation': r'consolidat',
    'ct_bronchiectasis': r'bronchiectas',
    'ct_septal_thickening': r'septal|interlobular',
}
GATE_RE = {k: re.compile(v, re.I) for k, v in GATES.items()}
# absent statements often name only the section ("Lymph nodes: Normal."): the header then counts as the term
HEADERS = {'mr_lymphadenopathy': r'lymph nodes?', 'ct_lymphadenopathy': r'lymph nodes?|mediastin',
           'mr_ascites': r'peritone', 'mr_hydronephrosis': r'kidneys?', 'mr_renal_calculus': r'kidneys?',
           'mr_gallstones': r'gallbladder', 'mr_gb_distension': r'gallbladder', 'mr_splenomegaly': r'spleen',
           'mr_hepatomegaly': r'liver', 'mr_fatty_liver': r'liver', 'mr_biliary_dilatation': r'liver|biliary',
           'mr_pancreatic_atrophy': r'pancreas', 'mr_pancreatic_duct': r'pancreas', 'mr_aortic_aneurysm': r'vasculature|aorta',
           'mr_prostatomegaly': r'prostate', 'ct_pericardial_effusion': r'heart|pericard', 'ct_cardiomegaly': r'heart'}
HEADER_RE = {k: re.compile(r'^\s*(' + v + r')[^:]{0,30}:', re.I) for k, v in HEADERS.items()}


def apply_gates(parts):
    keep = [(bool(GATE_RE[f].search(t)) or (p == 'absent' and f in HEADER_RE and bool(HEADER_RE[f].search(t))))
            if f in GATE_RE else True for f, t, p in zip(parts.finding, parts.text, parts.polarity)]
    parts = parts.copy()
    parts['finding_llm'] = parts.finding
    parts.loc[~np.array(keep), 'finding'] = 'other'
    # organ for vocabulary findings: from the finding and the sentence's side
    for i in np.where(parts.finding.values != 'other')[0]:
        orgs = FINDINGS[parts.finding.iat[i]][1]
        side = parts.side.iat[i]
        if len(orgs) == 2 and orgs[0].endswith('_left') and side in ('left', 'right'):
            parts.iat[i, parts.columns.get_loc('organ')] = [o for o in orgs if o.endswith('_' + side)][0]
        elif len(orgs) == 1:
            parts.iat[i, parts.columns.get_loc('organ')] = orgs[0]
    return parts


NEG_CUE = re.compile(r"\b(no|not|without|absent|none|negative|free of|nor|neither)\b", re.I)
CT_POST_NEG = re.compile(r"\b(was|were|is|are)\s+not\s+(observed|detected|seen|identified|present|found)|\bnot (observed|detected)\b", re.I)
# high-precision CT-RATE terms for sentences the LLM left as "other" (keyword fallback)
CT_FALLBACK = {'ct_hiatal_hernia': r'hiatal hernia|hiatus hernia', 'ct_emphysema': r'emphysem', 'ct_atelectasis': r'atelecta',
               'ct_bronchiectasis': r'bronchiectas', 'ct_pleural_effusion': r'pleural effusion', 'ct_pericardial_effusion': r'pericardial effusion',
               'ct_consolidation': r'consolidat', 'ct_cardiomegaly': r'cardiomegaly', 'ct_mosaic_attenuation': r'mosaic attenuation'}
CT_FALLBACK_RE = {k: re.compile(v, re.I) for k, v in CT_FALLBACK.items()}
ORGAN_WORDS = re.compile(r"\b(liver|hepatic|gallbladder|spleen|splenic|pancrea\w*|kidneys?|renal|colon|colonic|bowel|stomach|gastric|aort\w*|"
                         r"bladder|lungs?|pulmonary|heart|cardiac|esophag\w*|thyroid|adrenal|uterus|uterine|ovar\w*|prostat\w*|mediastin\w*|"
                         r"pleura\w*|trachea|bronch\w*|duodenum|appendix|pericard\w*|thorax|abdomen|pelvis|pelves|ureters?|mesenter\w*|"
                         r"hemithorax|diaphragm|ileum|jejunum|rectum|sigmoid|cecum|vertebra\w*|spine|adnexa\w*|sacr\w*|sternum|manubrium|ribs?|"
                         r"femur|femoral|psoas|muscles?|breasts?|axilla\w*|vena cava|svc|ivc|portal vein|veins?|arter\w*|sma|coronar\w*|"
                         r"epicardial|bile ducts?|cbd|l[1-5]|t1[0-2]|t[1-9])\b", re.I)


NORMAL_STMT = re.compile(r"\b(normal|patent|unremarkable|symmetric\w*|intact|preserved|natural|within normal limits|appears normal|"
                         r"non-?dilated|non-?enlarged|not (dilated|enlarged|visualized|seen|identified|detected))\b|^\W*no\b", re.I)


def _words(t):
    return sorted(re.findall(r"[a-z0-9]+", str(t).lower()))


def ok_negation(t, e):
    """LLM 'negated' rewrite: must differ, must add a negation, must not just append a vague 'no abnormality'."""
    if not isinstance(e, str) or not e.strip() or _words(e) == _words(t) or '[' in e:
        return False
    if re.search(r"no (evidence of )?(other |any )?abnormalit", e, re.I) and not re.search(r"abnormalit", t, re.I):
        return False
    if len(NEG_CUE.findall(e)) + len(CT_POST_NEG.findall(e)) <= len(NEG_CUE.findall(t)) + len(CT_POST_NEG.findall(t)):
        return False
    # negation appended after the unchanged sentence ("..., and there is no X present", "... and ABSENT"): unclear scope,
    # except the plain "<short finding> is absent"
    core = re.sub(r"[\W_]+$", "", t).lower()
    if e.lower().startswith(core):
        tail = e[len(core):]
        if not (re.fullmatch(r"\W*(is|are) absent\W*", tail, re.I) and len(_words(t)) <= 8):
            return False
    return True


def _organs(t):
    return {re.sub(r"s$", "", w.lower()) for w in ORGAN_WORDS.findall(t)}


def ok_swap(t, e):
    """LLM 'organ_swap' rewrite: must move the finding to a different organ or side, not paraphrase or annotate.
    Rejected: header-only changes ("Esophagus: <pleural effusion sentence>"), organs added while the original
    location is kept, and "both" -> one side."""
    if not isinstance(e, str) or not e.strip() or _words(e) == _words(t) or '[' in e or re.search(r"\bnote\b", e, re.I):
        return False
    if re.search(r"no (evidence of )?(other |any )?abnormalit", e, re.I) and not re.search(r"abnormalit", t, re.I):
        return False
    if ':' in t and ':' in e and t.split(':', 1)[1].strip().lower() == e.split(':', 1)[1].strip().lower() \
            and ORGAN_WORDS.search(t.split(':', 1)[1]):
        return False
    new_organ = _organs(e) - _organs(t)
    moved = bool(_organs(t) - _organs(e))
    side_flip = sorted(re.findall(r"\b(left|right)\b", e.lower())) != sorted(re.findall(r"\b(left|right)\b", t.lower()))
    if side_flip and not new_organ and re.search(r"\b(both|bilateral\w*)\b", t, re.I):
        return False      # "both lungs" -> "right lung" is still true
    return (bool(new_organ) and moved) or (side_flip and not new_organ)


def clean_claims(claims):
    """Post-hoc checks on LLM claims (after proof-reading 150 claims): Merlin finding/polarity follow the rule layer;
    CT-RATE keyword fallback for missed findings; rewrites that are not contradictions are dropped."""
    import organic.data.label_rules as lr
    c = claims.copy()
    c['polarity_llm'] = c.polarity
    c = c.sort_values(['volume', 'idx']).reset_index(drop=True)
    fnd, pol = c.finding.to_numpy(object).copy(), c.polarity.to_numpy(object).copy()
    mer = np.where(c.dataset.to_numpy() == 'merlin')[0]
    texts = c.text.to_numpy(object)
    vols = c.volume.to_numpy(object)
    # Merlin: rule layer decides polarity; a finding the rules do not see is dropped; missed findings are added
    start = 0
    while start < len(mer):
        end = start
        while end < len(mer) and vols[mer[end]] == vols[mer[start]]:
            end += 1
        rows = mer[start:end]
        secs = lr.sections([texts[r] for r in rows])
        for r, sec in zip(rows, secs):
            t = texts[r]
            if fnd[r] in lr.RULES:
                j = lr.judge(fnd[r], t, sec)
                if j is None:
                    if not (pol[r] == 'absent' and lr.normal_section_findings(t)):
                        fnd[r] = 'other'
                else:
                    pol[r] = j
            if fnd[r] == 'other':
                hits = [(f, j) for f in lr.RULES for j in [lr.judge(f, t, sec)] if j is not None]
                if len(hits) == 1:
                    fnd[r], pol[r] = hits[0]
                elif not hits and len(lr.normal_section_findings(t)) == 1:     # "Uterus: Normal."
                    fnd[r], pol[r] = lr.normal_section_findings(t)[0], 'absent'
        start = end
    # CT-RATE: keyword fallback for a single unambiguous term
    for r in np.where((c.dataset.to_numpy() == 'ctrate') & (fnd == 'other'))[0]:
        t = texts[r]
        hits = [f for f, rx in CT_FALLBACK_RE.items() if rx.search(t)]
        if len(hits) == 1:
            m = CT_FALLBACK_RE[hits[0]].search(t)
            neg = NEG_CUE.search(lr.clause_before(t, m.start())) or CT_POST_NEG.search(t[m.end():])
            fnd[r], pol[r] = hits[0], 'absent' if neg else ('uncertain' if lr.HEDGE.search(t) else 'present')
    c['finding'], c['polarity'] = fnd, pol
    # the LLM sometimes calls a normal statement "present" ("The splenic artery is patent"): no edits for those
    normal = [f == 'other' and bool(NORMAL_STMT.search(t)) for f, t in zip(c.finding, c.text)]
    keep_n = [ok_negation(t, e) and p == 'present' and not nm for t, e, p, nm in zip(c.text, c.neg_text, c.polarity, normal)]
    keep_s = [ok_swap(t, e) and p == 'present' and not nm for t, e, p, nm in zip(c.text, c.swap_text, c.polarity, normal)]
    print('rewrites kept: negated', round(float(np.mean([k for k, e in zip(keep_n, c.neg_text) if isinstance(e, str)])), 3),
          'organ_swap', round(float(np.mean([k for k, e in zip(keep_s, c.swap_text) if isinstance(e, str)])), 3))
    c.loc[~np.array(keep_n), 'neg_text'] = None
    c.loc[~np.array(keep_s), 'swap_text'] = None
    c['severity'] = [severity_of(t) for t in c.text]
    print('finding changed', round(float((c.finding != claims.sort_values(['volume', 'idx']).finding.values).mean()), 4),
          'polarity changed', round(float((c.polarity != c.polarity_llm).mean()), 4))
    return c


def merlin_labels(claims):
    """Hybrid patient labels for Merlin: LLM claims (gated, organ-checked) + rule layer (organic/data/label_rules.py).
    1 if any present claim; NaN if only uncertain; otherwise 0 (closed world, matching the closed-world test reference labels)."""
    import organic.data.label_rules as lr
    out = {}
    for vol, g in claims.groupby('volume', sort=False):
        g = g.sort_values('idx')
        texts = g.text.tolist()
        secs = lr.sections(texts)
        state = {f: set() for f in MERLIN_FINDINGS}
        organ_seen = set(g.organ)
        for t, sec, f_llm, pol in zip(texts, secs, g.finding, g.polarity):
            for f in MERLIN_FINDINGS:
                j = lr.judge(f, t, sec)
                if j:
                    state[f].add(j)
            if f_llm in state and pol in ('present', 'absent', 'uncertain'):
                # keep the LLM claim only if the rule layer agrees the finding is mentioned in organ context
                if lr.judge(f_llm, t, sec) is not None or pol == 'absent':
                    state[f_llm].add(pol)
            for f in lr.normal_section_findings(t):
                state[f].add('absent')
        row = {}
        for f in MERLIN_FINDINGS:
            st = state[f]
            orgs = set(FINDINGS[f][1])
            # closed world (as radiology reports and the test reference labels): not mentioned = absent; hedged = unlabelled
            if 'present' in st:
                row[f'label_{f}'] = 1.0
            elif 'uncertain' in st:
                row[f'label_{f}'] = np.nan
            else:
                row[f'label_{f}'] = 0.0
        out[vol] = row
    return pd.DataFrame.from_dict(out, orient='index')


def merge(cfg):
    out_dir = os.path.join(cfg.paths.work_root, 'parsed')
    df, uniq = build_jobs(cfg)
    parts = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{out_dir}/shard*.parquet'))], ignore_index=True)
    parts = apply_gates(parts)
    print('gate kept', round(float((parts.finding != 'other').sum() / max(1, (parts.finding_llm != 'other').sum())), 3),
          'of LLM vocabulary assignments')
    claims = df[['volume', 'dataset', 'rid']].merge(parts, on='rid')
    claims = clean_claims(claims)
    claims.drop(columns=['rid']).to_parquet(f'{out_dir}/claims.parquet', index=False)
    print('claims', len(claims), 'parsed share', round(float(parts.parsed.mean()), 4))
    print(parts.polarity.value_counts().to_dict())
    ml = merlin_labels(claims[claims.dataset == 'merlin']).rename_axis('volume').reset_index()
    ml.to_parquet(f'{out_dir}/merlin_labels.parquet', index=False)
    print('merlin labels', ml.shape, {k: float(np.nanmean(v)) for k, v in ml.drop(columns='volume').items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    cfg = load_env(a.env)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    if a.merge:
        return merge(cfg)
    if a.limit:
        global build_jobs
        _orig = build_jobs
        build_jobs = lambda c: (lambda d, u: (d, u.head(a.limit)))(*_orig(c))
    run_shard(cfg, a.shard, a.bs)


if __name__ == '__main__':
    main()
