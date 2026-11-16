"""Organ groups (TotalSegmentator v2 labels) and the unified finding vocabulary for Merlin + CT-RATE.

Organ ids are the anatomy-query groups of the model. Each finding names the organ groups whose erasure should
remove its evidence (empty = no mask; the finding gets only the invariance half of the evidence term).
"""

# organ group -> TotalSegmentator v2 "total" label ids
ORGANS = {
    'lung_left': [10, 11], 'lung_right': [12, 13, 14], 'heart': [51, 61], 'aorta': [52],
    'esophagus': [15], 'trachea': [16], 'liver': [5], 'gallbladder': [4], 'spleen': [1],
    'pancreas': [7], 'kidney_left': [3, 23], 'kidney_right': [2, 24], 'stomach': [6],
    'small_bowel': [18, 19], 'colon': [20], 'pelvic_organs': [21, 22],
}
ORGAN_NAMES = list(ORGANS)
ORGAN_ID = {o: i for i, o in enumerate(ORGAN_NAMES)}
BACKGROUND = 255

# organs that must never serve as an invariance control for each other (overlap or adjacency)
RELATED = {
    'liver': {'gallbladder'}, 'gallbladder': {'liver'}, 'stomach': {'esophagus'}, 'esophagus': {'stomach', 'aorta'},
    'small_bowel': {'colon'}, 'colon': {'small_bowel'}, 'lung_left': {'lung_right', 'heart'},
    'lung_right': {'lung_left', 'heart'}, 'heart': {'lung_left', 'lung_right', 'aorta'}, 'aorta': {'heart', 'esophagus'},
    'kidney_left': {'kidney_right'}, 'kidney_right': {'kidney_left'},
}

LUNGS = ['lung_left', 'lung_right']
KIDNEYS = ['kidney_left', 'kidney_right']

# name -> (dataset label column, organ groups, phrase for canonical prompts)
FINDINGS = {
    # CT-RATE (chest, 18 released labels)
    'ct_medical_material':        ('Medical material', [], 'medical material'),
    'ct_arterial_calcification':  ('Arterial wall calcification', ['aorta'], 'arterial wall calcification'),
    'ct_cardiomegaly':            ('Cardiomegaly', ['heart'], 'cardiomegaly'),
    'ct_pericardial_effusion':    ('Pericardial effusion', ['heart'], 'pericardial effusion'),
    'ct_coronary_calcification':  ('Coronary artery wall calcification', ['heart'], 'coronary artery calcification'),
    'ct_hiatal_hernia':           ('Hiatal hernia', ['esophagus', 'stomach'], 'hiatal hernia'),
    'ct_lymphadenopathy':         ('Lymphadenopathy', [], 'mediastinal lymphadenopathy'),
    'ct_emphysema':               ('Emphysema', LUNGS, 'emphysema'),
    'ct_atelectasis':             ('Atelectasis', LUNGS, 'atelectasis'),
    'ct_lung_nodule':             ('Lung nodule', LUNGS, 'lung nodule'),
    'ct_lung_opacity':            ('Lung opacity', LUNGS, 'lung opacity'),
    'ct_fibrotic_sequela':        ('Pulmonary fibrotic sequela', LUNGS, 'pulmonary fibrotic sequela'),
    'ct_pleural_effusion':        ('Pleural effusion', [], 'pleural effusion'),     # fluid lies outside the lung mask
    'ct_mosaic_attenuation':      ('Mosaic attenuation pattern', LUNGS, 'mosaic attenuation pattern'),
    'ct_peribronchial_thickening': ('Peribronchial thickening', LUNGS, 'peribronchial thickening'),
    'ct_consolidation':           ('Consolidation', LUNGS, 'consolidation'),
    'ct_bronchiectasis':          ('Bronchiectasis', LUNGS, 'bronchiectasis'),
    'ct_septal_thickening':       ('Interlobular septal thickening', LUNGS, 'interlobular septal thickening'),
    # Merlin (abdomen, 21 findings of the reference label set; training labels come from report parsing)
    'mr_hydronephrosis':          ('Kidney hydronephrosis', KIDNEYS, 'hydronephrosis'),
    'mr_renal_calculus':          ('Kidney renal calculus', KIDNEYS, 'renal calculus'),
    'mr_gallstones':              ('Gallbladder gallstone', ['gallbladder'], 'gallstones'),
    'mr_gb_distension':           ('Gallbladder distended', ['gallbladder'], 'gallbladder distension'),
    'mr_biliary_dilatation':      ('Liver biliary dilatation', ['liver'], 'biliary ductal dilatation'),
    'mr_hepatomegaly':            ('Liver hepatomegaly', ['liver'], 'hepatomegaly'),
    'mr_fatty_liver':             ('Liver fatty infiltration', ['liver'], 'hepatic steatosis'),
    'mr_hiatal_hernia':           ('Stomach hiatal hernia', ['esophagus', 'stomach'], 'hiatal hernia'),
    'mr_pancreatic_atrophy':      ('Pancreas Atrophy', ['pancreas'], 'pancreatic atrophy'),
    'mr_pancreatic_duct':         ('Pancreas duct dilation', ['pancreas'], 'pancreatic duct dilatation'),
    'mr_splenomegaly':            ('Spleen splenomegaly', ['spleen'], 'splenomegaly'),
    'mr_appendicitis':            ('Bowel Appendicitis', ['colon'], 'appendicitis'),
    'mr_diverticulosis':          ('Bowel diverticulosis', ['colon'], 'colonic diverticulosis'),
    'mr_bowel_obstruction':       ('Bowel obstruction', ['small_bowel'], 'bowel obstruction'),
    'mr_aortic_aneurysm':         ('Abdominal_Aorta Aneurysm', ['aorta'], 'abdominal aortic aneurysm'),
    'mr_aortic_stent':            ('Abdominal_Aorta endovascular stent', ['aorta'], 'abdominal aortic endovascular stent'),
    'mr_prostatomegaly':          ('Prostate prostatomegaly', ['pelvic_organs'], 'prostatomegaly'),
    'mr_ascites':                 ('Abdomen ascites/Abnormal abdominal fluid', [], 'ascites'),
    'mr_lymphadenopathy':         ('lymphadenopathy', [], 'abdominal lymphadenopathy'),
    'mr_uterine_fibroid':         ('Uterus Fibroid', [], 'uterine fibroid'),
    'mr_ovarian_cyst':            ('Ovary Cyst', [], 'ovarian cyst'),
}
FINDING_NAMES = list(FINDINGS)
# Size/shape findings: erasing the organ keeps its outline, so the erased scan still shows the size. Their sensitivity
# erasure is not a valid evidence test (reported as a limitation); aug.ev_skip_size leaves them out of the erase terms.
SIZE_FINDINGS = ['ct_cardiomegaly', 'ct_hiatal_hernia', 'mr_hepatomegaly', 'mr_splenomegaly', 'mr_prostatomegaly',
                 'mr_gb_distension', 'mr_pancreatic_atrophy', 'mr_aortic_aneurysm', 'mr_bowel_obstruction', 'mr_hiatal_hernia']
FINDING_ID = {f: i for i, f in enumerate(FINDING_NAMES)}
CTRATE_FINDINGS = [f for f in FINDING_NAMES if f.startswith('ct_')]
MERLIN_FINDINGS = [f for f in FINDING_NAMES if f.startswith('mr_')]
# finding family (label prefix) of each dataset: chest sets answer the CT-RATE findings, abdominal sets the Merlin ones
FAMILY = {'ctrate': 'ct_', 'radchest': 'ct_', 'merlin': 'mr_'}
REGION = {'ctrate': 'chest', 'radchest': 'chest', 'merlin': 'abdominal'}


def finding_organ_ids(f):
    return [ORGAN_ID[o] for o in FINDINGS[f][1]]


def control_organ_ids(f):
    """Organs that are neither the finding's target nor related to it (invariance controls)."""
    tgt = set(FINDINGS[f][1])
    rel = set().union(*[RELATED.get(o, set()) for o in tgt]) if tgt else set()
    return [ORGAN_ID[o] for o in ORGAN_NAMES if o not in tgt and o not in rel]


def canonical_prompts(f):
    """(positive, negative) report-style sentences used for the evidence logit when no report sentence applies."""
    p = FINDINGS[f][2]
    return f'There is {p}.', f'No {p}.'
