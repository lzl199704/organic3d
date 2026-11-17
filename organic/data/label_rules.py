"""Rule layer on top of the LLM sentence claims (Merlin findings). Written after proof-reading 70 Merlin reports;
its accuracy is reported on a separate, freshly proof-read set.

For each sentence and finding it returns 'present', 'absent', 'uncertain' or None (not mentioned), using
  - a finding pattern (strong terms), required organ context for organ-ambiguous terms (sentence or its section),
  - negation cues before the term in the same clause (NegEx-style), and hedge / trace cues -> 'uncertain'.
Labelling policy (training): uncertain -> NaN, so borderline statements never become positives or negatives.
"""
import re

NEG = re.compile(r"\b(no|not|without|negative for|free of|absence of|absent|resolved|resolution of|repair of|rather than|"
                 r"no evidence of|no ct evidence of|no radiographic|none|nor|with out|no longer)\b", re.I)
HEDGE = re.compile(r"\b(may|might|possible|possibly|cannot be excluded|can not be excluded|questionable|equivocal|"
                   r"borderline|versus|vs\.?|could|suggestion of|less likely|indeterminate|differential)\b", re.I)
TRACE = re.compile(r"\b(trace|minimal|tiny|physiolog\w*|small amount of nonspecific|nonspecific (\w+ )?free fluid)\b", re.I)

# finding -> (term pattern, organ-context pattern or None, extra rule)
RULES = {
    'mr_hydronephrosis':     (r'hydroneph\w*|hydroureteroneph\w*|pelvocaliectasis|pelviectasis|caliectasis|(dilat\w*|distent\w*|distens\w*)\W+(?:\w+\W+){0,4}?(renal pelvis|renal pelves|collecting systems?|calyces)|(renal pelvis|collecting systems?|calyces)\W+(?:\w+\W+){0,3}?(dilat\w*|distend\w*)|fullness\W+(?:\w+\W+){0,4}?(collecting systems?|renal pelv\w*)', None, 'hydro'),
    'mr_renal_calculus':     (r'nephrolithiasis|nephroliths?|staghorn\w*|(renal|kidneys?|pole|interpolar|calyx|calyceal|ureteral|ureteropelvic|ureterovesicular|non[- ]?obstruct\w*)\W+(?:\w+\W+){0,6}?(calcul\w*|stones?)|(calcul\w*|stones?)\W+(?:\w+\W+){0,6}?(kidneys?|renal|pole|interpolar)', r'kidney|renal|ureter|pole|interpolar|nonobstruct|non-obstruct', 'renal'),
    'mr_gallstones':         (r'gallstones?|chole?lith\w*|cholelithasis|\b(stones?|calcul\w*)', None, 'gb_stone'),
    'mr_gb_distension':      (r'distend\w*|distension|distention|hydrop\w*|dilated|dilatation|enlarged', r'gallbladder', 'gb'),
    'mr_biliary_dilatation': (r'(biliary|bile duct|ductal|\bcbd\b|common (bile|hepatic) duct|intrahepatic|extrahepatic)\W+(?:\w+\W+){0,6}?dilat\w*|dilat\w*\W+(?:\w+\W+){0,4}?(biliary|bile duct|\bcbd\b|common (bile|hepatic) duct)|periductal dilat\w*', None, None),
    'mr_hepatomegaly':       (r'hepatomegaly|hepatosplenomegaly|liver\W+(?:\w+\W+){0,4}?(enlarged|enlargement)|enlarged liver', None, None),
    'mr_fatty_liver':        (r'focal fat\w*|fat(ty)? sparing|steatosis|hepatosteatosis|fatty liver|fatty infiltration of the liver|diffuse(ly)? (fatty infiltration|decreased attenuation|hypoattenuat\w*|low attenuation|hypodense)|(mild|moderate|severe)?\s*diffuse fatty infiltration|liver is (diffusely |mildly )?(hypoattenuat\w*|hypodense|low (in )?attenuation|decreased in attenuation)|decreased attenuation of the liver', r'liver|hepat', 'liver'),
    'mr_hiatal_hernia':      (r'hiatal hernia|hiatus hernia', None, None),
    'mr_pancreatic_atrophy': (r'atroph\w*|fatty replacement|lipomatosis|fatty infiltration', r'pancrea', 'pancreas'),
    'mr_pancreatic_duct':    (r'pancreatic duct\W+(?:\w+\W+){0,6}?(dilat\w*|prominen\w*)|(dilat\w*|prominen\w*)\W+(?:\w+\W+){0,6}?pancreatic duct|pancreatic ductal (dilat\w*|prominen\w*)|pancreatic (and|&) (biliary|bile) ductal dilat\w*|(biliary|bile) and pancreatic ductal dilat\w*|pancreatic duct\W+(?:\w+\W+){0,3}?measur\w*\W+(?:up to\W+)?\d+(\.\d)? ?mm', None, 'pduct'),
    'mr_splenomegaly':       (r'splenomegaly|hepatosplenomegaly|spleen\W+(?:\w+\W+){0,6}?(enlarged|enlargement)|enlarged spleen|stable mild enlargement', r'splen|spleen', 'spleen'),
    'mr_appendicitis':       (r'appendicitis|(dilated|thickened|inflamed|perforated|fluid[- ]filled)\W+(?:\w+\W+){0,2}?appendix|appendix\W+(?:\w+\W+){0,5}?(dilated|inflamed|perforat\w*|thickened|mural thickening|wall thickening|hyperemi\w*)|periappendiceal (abscess|inflammation|inflammatory|stranding|phlegmon)', None, 'appendix'),
    'mr_diverticulosis':     (r'diverticulosis|diverticul\w*', r'colon|colonic|sigmoid|cecal|cecum|descending|ascending|transverse|pancolonic|gastrointestinal|bowel|diverticulosis', 'colon'),
    'mr_bowel_obstruction':  (r'(small bowel|bowel|closed[- ]loop|mechanical|partial|low[- ]grade|high[- ]grade|intestinal|colonic|functional)\W+(?:\w+\W+){0,2}?obstruction|\bsbo\b|(duodenum|bowel|colon)\W+(?:\w+\W+){0,6}?obstructed|transition point', None, 'sbo'),
    'mr_aortic_aneurysm':    (r'\w*aneurysm\w*', r'aort|iliac', 'aneurysm'),
    'mr_aortic_stent':       (r'endograft|\bevar\b|(aortic|endovascular) stent|stent[- ]?graft|endovascular (aneurysm )?repair|aorto\w*\W+(?:\w+\W+){0,3}?(stent|graft)|iliac (artery )?stent|stent\w*\W+(?:\w+\W+){0,4}?iliac', None, 'stent'),
    'mr_prostatomegaly':     (r'prostatomegaly|prostate\W+(?:\w+\W+){0,4}?(enlarged|enlargement)|enlarged prostate|prostat\w* (hypertrophy|hyperplasia)', None, None),
    'mr_ascites':            (r'ascites|free fluid|intraperitoneal fluid|abdominal fluid|pelvic fluid|fluid in the pelvis', None, 'ascites'),
    'mr_lymphadenopathy':    (r'lymphadenopathy|adenopathy|(enlarged|large|bulky|necrotic)\W+(?:\w+\W+){0,4}?(lymph )?nodes?|nodes?\W+(?:\w+\W+){0,8}?(\d\d) ?mm|nodes?\W+(?:\w+\W+){0,8}?[1-9](\.\d)? ?cm', None, 'nodes'),
    'mr_uterine_fibroid':    (r'fibroids?|leiomyoma\w*|myomatous', None, None),
    'mr_ovarian_cyst':       (r'(ovar\w*|adnex\w*)\W+(?:\w+\W+){0,6}?cyst\w*|cyst\w*\W+(?:\w+\W+){0,6}?(ovar\w*|adnex\w*)|corpus lute\w* cyst|dermoid|corpus lute\w*', None, 'ovary'),
}
RULE_RE = {f: (re.compile(p, re.I), re.compile(c, re.I) if c else None, x) for f, (p, c, x) in RULES.items()}
ORGAN_NORMAL = {   # section header -> findings set to absent when the section reads "Normal"/"Unremarkable"
    r'liver': ['mr_hepatomegaly', 'mr_fatty_liver', 'mr_biliary_dilatation'], r'gallbladder': ['mr_gallstones', 'mr_gb_distension'],
    r'spleen': ['mr_splenomegaly'], r'pancreas': ['mr_pancreatic_atrophy', 'mr_pancreatic_duct'],
    r'kidneys?': ['mr_hydronephrosis', 'mr_renal_calculus'], r'lymph nodes?': ['mr_lymphadenopathy'],
    r'peritone\w*': ['mr_ascites'], r'prostate': ['mr_prostatomegaly'], r'uterus': ['mr_uterine_fibroid'],
    r'ovar\w*|uterus and ovaries': ['mr_ovarian_cyst'], r'vasculature|aorta': ['mr_aortic_aneurysm', 'mr_aortic_stent'],
    r'(gastrointestinal|bowel)': ['mr_bowel_obstruction', 'mr_appendicitis', 'mr_diverticulosis'],
}
NORMAL_RE = re.compile(r'^\s*([A-Za-z ,/&]+?)\s*:\s*(normal|unremarkable|within normal limits|no abnormalit\w*)\s*\.?\s*$', re.I)
HEADER_RE = re.compile(r'^\s*([A-Z][A-Za-z ,/&]{1,40}?)\s*:')


PSEUDO_NEG = re.compile(r"\b(no (significant |interval |substantial )?change|not significantly changed|no new|without change)\b", re.I)
POST_NEG = re.compile(r"^\W*(?:\w+\W+){0,6}?(has|have|had)?\s*(resolved|been resolved|decompressed|repaired|no longer (present|seen|visualized|identified|evident))\b", re.I)
# the finding is the reason for the exam, a past event or a request, not an observation on this scan
NOT_OBSERVED = re.compile(r"correlate (for|with) (symptoms|clinical)|evaluate for|rule out|\br/o\b|on the prior (study|exam|ct|scan|imaging)|"
                          r"(was|were) (likely )?responsible for|(prior|previous|recent|outside) (ultrasound|us|mri|ct|exam|study)\W+(?:\w+\W+){0,2}?(demonstrated|showed|seen)|"
                          r"seen on (the )?(recent|prior|outside|same[- ]day) (ultrasound|us|mri)|(sonographic|ultrasound) findings|evaluation for", re.I)
# the finding is judged unlikely after the term ("appendicitis is felt unlikely", "obstruction not favored")
POST_UNLIKELY = re.compile(r"^\W*(?:\w+\W+){0,4}?(unlikely|not favou?red|not suspected|is not present)\b", re.I)
PRE_UNLIKELY = re.compile(r"\b(makes?|making|render\w*)\W+(?:\w+\W+){0,3}$", re.I)
HISTORY = re.compile(r"\b(prior|previous|previously|history of|status post|s/p|old)\W+(?:\w+\W+){0,2}$", re.I)
LOCAL_FLUID = re.compile(r"\bperi\w+\s+(free\s+)?$", re.I)
OTHER_ARTERY = re.compile(r"iliac|femoral|splenic|renal|hepatic|mesenteric|celiac|popliteal|gastroduodenal|pancreaticoduodenal|carotid", re.I)
TERMINATE = re.compile(r"\b(but|however|although|though|except|which|whereas)\b|[.;]|,\s*(representing|consistent with|compatible with|likely|suggesting|reflecting)\b|"
                       r"\bwith (stable|persistent|mild|moderate|severe|new|interval|marked)\b", re.I)


def clause_before(text, start, window=120):
    """NegEx-style scope: text before the term back to a sentence break or termination word (commas do not end it)."""
    pre = re.sub(r'\b(e\.g|i\.e)\.', 'eg,', text[max(0, start - window):start], flags=re.I)
    last = None
    for m in TERMINATE.finditer(pre):
        last = m
    if last is not None:
        pre = pre[last.end():]
    k = pre.rfind(':')                      # section header ends the scope too
    return pre[k + 1:] if k >= 0 else pre


_RANK = {None: 0, 'absent': 1, 'uncertain': 2, 'present': 3}


def judge(finding, text, section):
    """Judge every mention of the finding in the sentence; present > uncertain > absent > not mentioned."""
    pat = RULE_RE[finding][0]
    best = None
    for m in pat.finditer(text):
        if ':' in m.group(0):  # match runs across a section header: prefer a match after it, else keep (header as subject)
            m = pat.search(text, m.start() + m.group(0).rindex(':') + 1) or m
        j = _judge_match(finding, text, section, m)
        if _RANK[j] > _RANK[best]:
            best = j
    return best


def _judge_match(finding, text, section, m):
    pat, ctx, extra = RULE_RE[finding]
    if NOT_OBSERVED.search(text):
        return None
    near = clause_before(text, m.start(), 80) + m.group(0) + re.split(r'[.;]', text[m.end():m.end() + 50])[0]
    if ctx is not None and not (ctx.search(text) or (section and ctx.search(section))):
        return None
    if extra == 'liver' and re.search(r'mass|tumou?r|carcinoma|\bhcc\b|metasta', text, re.I) and not re.search(r'steatosis|fatty', text, re.I):
        return None
    if extra == 'liver' and re.search(r'focal fat|fat(ty)? sparing|focal steatosis|geographic', text, re.I) \
            and not re.search(r'(hepatic|diffuse\w*|background|mild|moderate|severe) (hepatic )?steatosis|hepatosteatosis|fatty liver|diffuse', text, re.I):
        # focal fat is not diffuse fatty liver, but the test labels count it: leave unlabelled rather than negative
        return 'absent' if NEG.search(clause_before(text, m.start())) else 'uncertain'
    if extra == 'ovary' and not re.search(r'\bcysts?\b', m.group(0), re.I) and re.search(r'(represent\w*|consistent with|compatible with|likely|favou?r\w*)\W+(?:\w+\W+){0,3}?(physiolog\w* |normal |dominant )?(ovarian )?follic', text, re.I):
        return None
    if extra == 'ovary' and re.search(r'nabothian|renal|kidney|hepatic|liver|pancrea|splen', m.group(0), re.I) and not re.search(r'corpus lute|(ovar\w*|adnex\w*) cyst', m.group(0), re.I):
        return None
    if extra == 'aneurysm' and re.match(r'non-?aneurysm', m.group(0), re.I):
        return 'absent'
    if extra == 'aneurysm':
        if not re.search(r'aort', near, re.I):
            return ('uncertain' if re.search(r'iliac', near, re.I) else None) if not NEG.search(clause_before(text, m.start())) else None
        if re.search(r'pseudoaneurysm', m.group(0), re.I):
            return 'uncertain'
    if extra == 'stent':
        if re.search(r'thoracic|aort|endograft|evar|endovascular', near, re.I) is None:
            return 'uncertain' if re.search(r'iliac', near, re.I) else None
    if extra == 'renal' and not re.search(r'kidney|renal|pole|interpolar|calyx|calyceal|nephrolith|staghorn', m.group(0), re.I) \
            and not re.search(r'kidney|renal|pole|interpolar', text + ' ' + (section or ''), re.I):
        return 'absent' if NEG.search(clause_before(text, m.start())) else 'uncertain'           # ureteral stone only: the test labels are inconsistent about it
    if extra == 'gb_stone' and not re.search(r'gallstone|chole?lith', m.group(0), re.I):
        # plain "stones"/"calculi": the sentence (or its section) must be about the gallbladder, not kidney or bile duct
        if re.search(r'kidney|renal|ureter|nephro|\bcbd\b|common bile duct|choledoch|bladder(?!.*gall)|pancrea', text, re.I) \
                and not re.search(r'gallbladder', text, re.I):
            return None
        if not (re.search(r'gallbladder', text, re.I) or (section and re.search(r'gallbladder', section, re.I))):
            return None
    if extra == 'gb':
        if not re.search(r'gallbladder', text, re.I) and re.search(r'bladder|bowel|stomach|colon|loops?|duct|biliary|pelvis|ureter|vein|aort', text, re.I):
            return None
        if not re.search(r'gallbladder', near, re.I) and not (section and re.search(r'gallbladder', section, re.I)):
            return None
        if re.search(r'incompletely|under-?distended|collapsed|contracted|nondistended|non-distended', near, re.I):
            return 'absent'
    if extra == 'appendix' and not re.search(r'appendicitis', m.group(0), re.I):
        m = re.search(r'appendicitis', text, re.I) or m      # the diagnosis, when stated, decides
    if extra == 'hydro' and re.search(r'fullness', m.group(0), re.I):
        return 'absent' if NEG.search(clause_before(text, m.start())) else 'uncertain'
    if extra == 'sbo' and re.search(r'transition point', m.group(0), re.I) and re.search(r'ureter|ureteropelvic|biliary|duct|vein|artery', near, re.I):
        return None
    if extra == 'pancreas' and re.search(r'kidney|renal|spleen|splenic|liver|hepat|testic|muscle|thyroid|adrenal', near, re.I) and not re.search(r'pancrea', near, re.I):
        return None
    if extra == 'hydro' and re.search(r'extrarenal', text, re.I) and not re.search(r'hydroneph|caliectasis|pelvocaliectasis', text, re.I):
        return 'uncertain'
    if extra == 'ascites' and LOCAL_FLUID.search(text[:m.start()]) and not re.search(r'ascites', text, re.I):
        return None
    if extra == 'colon' and re.search(r'duodenal diverticul', text, re.I) and not re.search(r'colon|sigmoid|diverticulosis', text, re.I):
        return None
    before = PSEUDO_NEG.sub(' ', clause_before(text, m.start()))
    if POST_NEG.search(text[m.end():]):
        return 'absent'
    if re.search(r'\b(mostly|partially|nearly|largely) resolved', text[m.end():], re.I):
        return 'uncertain'
    if POST_UNLIKELY.search(text[m.end():]) or PRE_UNLIKELY.search(before) and re.search(r'unlikely', text[m.end():m.end() + 20], re.I):
        return 'absent'
    if extra in ('appendix', 'sbo', 'gb_stone') and HISTORY.search(before) and not re.search(r'persist|again|remain|redemonstrat|stable|unchanged', text, re.I):
        return None
    if extra == 'ascites' and re.search(r'collection', text, re.I) and not re.search(r'ascites|free fluid', text, re.I):
        return None
    if extra == 'nodes' and m.lastindex and re.search(r'subcentimeter|sub-centimeter', text, re.I) and not re.search(r'lymphadenopathy|enlarged', text, re.I):
        return 'uncertain'
    if NEG.search(re.sub(r'non[- ]?obstruct\w*', ' ', m.group(0), flags=re.I)) and not re.search(r'\bno (change|new)\b', m.group(0), re.I):
        return 'absent'
    if extra == 'sbo' and re.search(r'gastric outlet', m.group(0), re.I):
        return None
    if extra == 'sbo' and re.search(r'functional', m.group(0), re.I):
        return 'uncertain'
    if extra == 'pduct' and re.search(r'measur', m.group(0), re.I) and not re.search(r'dilat', m.group(0), re.I):
        return 'present' if float(re.findall(r'(\d+(?:\.\d)?) ?mm', m.group(0))[-1]) >= 4 else None
    if extra == 'pduct' and re.search(r'(decreas\w*|improv\w*|reduc\w*)\W+(?:\w+\W+){0,5}?$', text[:m.start()], re.I):
        return 'uncertain'
    if extra == 'ovary' and not re.search(r'cyst', m.group(0), re.I):
        return 'uncertain'     # dermoid / corpus luteum named without "cyst"
    if extra == 'liver' and re.search(r'diffuse\w* (low|decreased|hypo)', m.group(0), re.I) and re.search(r'cirrho|edema|congest|infiltrat|perfusion', text, re.I) \
            and not re.search(r'steatosis|fatty', text, re.I):
        return 'uncertain'
    if extra == 'sbo' and re.search(r'interval (reduction|decrease|improvement)|improv\w*|resolving', text, re.I):
        return 'uncertain'
    if extra == 'pduct' and not re.search(r'dilat', m.group(0), re.I) and not NEG.search(clause_before(text, m.start())):
        return 'uncertain'     # "prominent" duct without stated dilatation (same policy as prominent nodes)
    if extra == 'pduct' and re.search(r'without (frank|significant|definite|true) (\w+ )?dilat', text, re.I):
        return 'uncertain'
    if extra == 'pduct' and re.search(r'\b(air|gas|stent)\b', text, re.I) and not re.search(r'dilat\w*|prominen\w*', text[m.start():], re.I):
        return None
    if re.search(r'\b(not|non|no)\b', re.sub(r'non[- ]?obstruct\w*', ' ', m.group(0), flags=re.I), re.I) \
            or (text[max(0, m.start() - 4):m.start()].lower().endswith(('non', 'non-', 'un')) and not m.group(0).lower().startswith('obstruct')):
        return 'absent'
    if NEG.search(before) or re.search(r'(not (meet|reach)|none (is|are)|no .{0,20}pathologic)', text[m.start():], re.I) and extra == 'nodes':
        return 'absent'
    if extra == 'ascites' and TRACE.search(text):
        return 'uncertain'
    if extra == 'nodes' and re.search(r'prominent|borderline|subcentimeter|sub-centimeter|reactive|not enlarged|within normal', text, re.I) \
            and not re.search(r'lymphadenopathy|adenopathy|enlarged', text, re.I):
        return 'uncertain'
    if extra == 'pancreas' and not re.search(r'atroph', m.group(0), re.I):
        return 'uncertain' if re.search(r'pancrea', near, re.I) or (section and re.search(r'pancrea', section, re.I)) else None
    if extra == 'appendix' and not re.search(r'appendicitis', m.group(0), re.I):
        return 'uncertain'
    if extra == 'gb_stone' and re.search(r'\bor (polyps?|sludge|wall calcification)', text[m.end():m.end() + 30], re.I):
        return 'uncertain'   # abnormal appendix described without the diagnosis (tumour, IBD, reactive): unlabelled
    if HEDGE.search(before) or re.search(r'possible|cannot be excluded|not excluded|is suspected|are suspected', text[m.end():m.end() + 40], re.I):
        return 'uncertain'
    return 'present'


def sections(sentences):
    """Section header in force for each sentence (templated reports: 'Liver and biliary tree: ...')."""
    cur, out = None, []
    for s in sentences:
        h = HEADER_RE.match(s)
        if h and not h.group(1).isupper() or (h and len(h.group(1)) < 25):
            cur = h.group(1)
        out.append(cur)
    return out


def normal_section_findings(sentence):
    m = NORMAL_RE.match(sentence)
    if not m:
        return []
    head = m.group(1).lower()
    out = []
    for pat, fs in ORGAN_NORMAL.items():
        if re.fullmatch(r'.*\b(' + pat + r')\b.*', head):
            out += fs
    return out
