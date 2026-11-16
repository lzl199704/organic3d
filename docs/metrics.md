# Metric definitions

The definitions of the paper's Section 3.4, grouped by the four tasks (classification, clinical finding independence, organ
dependence, laterality), with the paper's metric names: MLC-AUROC, MP, OE, OEH, LE, Side, LL and IA-F1. Each one names the code that computes it. Notation:
$A_f(g)$ is the AUROC of a per-scan score $g$ against the label $y_f$ of finding $f$; $g=\Delta_f$ for encoders and
$g=p_f$ for VLMs. **Every metric is computed per finding and reported as the mean over findings** (each finding counts
equally; e.g. MLC-AUROC on CT-RATE is the mean of the 18 per-finding AUROCs). Findings need at least 10 positives in the
test set to be scored. All metrics are oriented so that higher is better.

## Scores

**Encoders.** For an encoder with image–sentence similarity $S(V,s)$, the score of finding $f$ is

$$\Delta_f(V)=S(V,s_f^{+})-S(V,s_f^{-}),$$

the similarity to the positive prompt minus that to the negative prompt. **Every released encoder is scored with its
own published prompt template and score rule**: $\Delta_f$ ranks scans exactly as the model's own two-prompt softmax
(CT-CLIP, CT-VocabFine, Merlin, HLIP, fVLM, ViSD-Boost; RADAR+ over its prompt ensembles), equals ItemizedCLIP's
positive-only score (its negative is a null text) and Jolia's mean over its eight template pairs. Models without a
published zero-shot protocol (COLIPRI, SPECTRE, ORGANIC-3D) use "There is $f$." / "No $f$.". Code: `extaudit/core.py`
`Model.deltas`; each adapter's `prompts()` in `extaudit/models/`; findings outside the label list (tumour findings) in
`extaudit/native_prompts.py`.

**VLMs.** $p_f(V)$ is the probability of "Yes" against "No" at the first answer token of a yes/no question about $f$
(the yes/no logit difference when the adapter provides it; same ranking).

## Classification

- **MLC-AUROC** $=A_f(\Delta_f)$ on the report-derived labels of CT-RATE, RAD-ChestCT and Merlin
  (`scripts/extaudit_metrics.py`, summary key `zero_shot`).
- **Lesion AUROC** on labels from expert masks:
  - ReXGroundingCT findings on its 464 radiologist-annotated scans (a subset of the CT-RATE test set whose labels come
    from the lesion masks) (`scripts/lesion_cls_metrics.py`);
  - **tumour detection on AbdomenAtlas 3.0, zero-shot for every encoder**, with each model's own prompts for liver
    mass, pancreatic mass and renal mass (`scripts/lesion_det_metrics.py`, task `aa3_det`): IID test split, 926 scans,
    radiologist-revised tumour masks; each organ is a finding, positive with a tumour of at least 10 mm and negative
    with no lesion; the dataset value is the **mean over the three tumour findings** (liver, pancreas, kidney).

    The free sentence pair "There is (no) mass in the [organ]." and the positive prompt alone are reported as checks.
- **VLMs**: answer AUROC $A_f(p_f)$; balanced accuracy and F1 of the stated answer (Yes when $p_f\geq0.5$)
  (`scripts/vlm_metrics.py`, `scripts/main_tables.py`).

## Clinical finding independence

A matched pair $(i,j)\in\Pi_f$ has $y_f(V_i)=1$, $y_f(V_j)=0$ and identical labels on every other labelled finding
(missing labels must match too). The matched-pair AUROC (MP)

$$\mathrm{MP}_f=\frac{1}{|\Pi_f|}\sum_{(i,j)\in\Pi_f}\Big(\mathbb 1[g(V_i)>g(V_j)]+\tfrac12\mathbb 1[g(V_i)=g(V_j)]\Big)$$

is a covariate-adjusted AUROC whose covariates are all other labelled findings; a model that predicts $f$ from the
findings that usually accompany it scores 0.5. Same definition for encoders and VLMs. Code: `organic/ladder.py`
`matched_pair_auroc` (summary key `matched_pair_auroc`).

## Organ dependence

**Encoders: organ-erasure ΔAUROC (OE).** With fill $c$ (fat in the tables; soft tissue, and the unseen water, median
and blur fills, in the supplement),

$$\mathrm{OE}_f=\operatorname*{mean}_{U\subseteq\mathcal C(f)}A_f(g\circ\mathcal E^{c}_{U})-\operatorname*{mean}_{U\cap O(f)\neq\varnothing}A_f(g\circ\mathcal E^{c}_{U}),$$

the AUROC points lost when the finding's organ is erased minus those lost when a control organ is erased, each term on
the scans that contain $U$. $O(f)$ are the target organs, $\mathcal C(f)$ the control organs (`organic/vocab.py`).
The 10 size and shape findings are excluded, because erasure keeps the organ's outline. Summary key
`effect_<fill>_texture`, stored with the opposite sign (OE $=-$`effect_fat_texture`). Code:
`organic/evaluate_stageA.py` `metrics`, `write_outputs`.

**VLMs: organ-erasure hit rate (OEH).** Among findings stated correctly on the clean scan ($y_f=1$, $p_f\ge t_f$, thresholds
cross-fitted over patient folds), the share no longer stated after a target-organ erasure minus that after a control
erasure (`scripts/report_tracks.py`, key `retraction`; its parts and the unseen fills: `scripts/oeh_breakdown.py`). The
OE of a VLM is the same ΔAUROC as for encoders, computed on $p_f$.

**Lesion-erasure hit rate (LE).** The share of expert-segmented lesions whose replacement by the organ's normal tissue
(median HU of the organ outside every lesion, plus noise) lowers the lesion's sentence score more than the same edit of a
same-shaped healthy region of the same organ (chance 0.5); ReXGroundingCT lung findings and AbdomenAtlas 3.0 liver,
pancreas and kidney tumours. No training step uses this edit. Sets: `scripts/prepare_heal.py`; scores:
`LE_SET=<rex|aa3> python -m extaudit.core --task lesion_erasure`; metric: `scripts/heal_metrics.py`.

## Laterality (encoders)

- **Laterality accuracy (Side)**: the share of report sentences that name one side and score above the same sentence
  with the side swapped, $S(V,s)>S(V,s^{\leftrightarrow})$, on the original scan, balanced over left- and right-side
  sentences (chance 0.5, also for a model that ignores side). CT-RATE and Merlin test reports. Scores:
  `extaudit/side.py` (sentence pairs from `scripts/prepare_side.py`); metric: `scripts/side_metrics.py`.
- **Mirror-consistent laterality** (supplementary, not in the paper's tables): for a sentence $s$ that names one side and its side-swapped version
  $s^{\leftrightarrow}$, the share with $S(V,s)>S(V,s^{\leftrightarrow})$ **and**
  $S(\mathcal XV,s^{\leftrightarrow})>S(\mathcal XV,s)$ on the left–right mirrored scan (chance 0.25; 0 for a model that
  takes the side from the wording). Sentences: CT-RATE and Merlin test reports. Summary key `lat_both`.
- **Lung lobe erasure hit rate (LL)**: the share of CT-RATE test sentences naming one lung lobe for which erasing that lobe
  lowers the sentence's score more than erasing any of the other four (chance 0.20). Code: `extaudit/chest.py`,
  `scripts/chest_harness_metrics.py`.

Laterality is not a VLM task: the yes/no questions come from the side-free label list.

## Reports (VLMs)

Free-text reports are scored against the findings section of the reference report.

- **Finding labels**: labels extracted from the generated report (CT-RATE: the RadBERT classifier of the CT-RATE
  protocol; Merlin: our rule-based labeller) against the reference labels: F1, precision and recall on positive
  findings, and CRG (`scripts/report_metrics_std.py`).
- **Agreement with the reference report**: GREEN (an LLM judge, run with a **4,096-token window** because CT report
  pairs overflow its 2,048-token default; `scripts/report_green.py --max_length 4096`), RadGraph F1 and BLEU-4
  (`scripts/report_metrics_std.py`).
- **Finding-based reports** state every finding with $p_f\ge t_f$, thresholds cross-fitted over five patient folds
  for every model alike; image-attributable F1 (IA-F1) $=\mathrm{F1}-\mathrm{F1}_{\mathrm{wr}}$ subtracts the F1 of another
  patient's report (`scripts/report_tracks.py`).

Our VLM writes answer-first reports; at inference the report is started with an anatomy prefix, the first item of the
dataset's findings order (`REPORT_HEAD_MERLIN='Lower thorax:'`, `REPORT_HEAD_CTRATE='Trachea'`, read by
`extaudit/vlm_models/organic3d_vlm.py`).

## Evidence maps (qualitative, Figure 3)

Occlusion sensitivity: a 24 × 24 × 32 mm block is slid over the axial slice through a lesion (16 mm stride) and
replaced by normal-looking tissue (median lung density in the lung, fat elsewhere, plus noise); the map records the drop
in the model's score for the finding's sentence; each map is scaled to its own maximum. Task `occlusion` (`extaudit/occlusion.py`; case table via
`$FIG4_CASES`).

## Supplementary metrics

Density-edit specificity (`scripts/chest_harness_metrics.py`) and the anatomy-token attention analysis
(`scripts/slot_attention_maps.py`, `scripts/summarise_slot_attention.py`).

## Statistics

- 95% percentile intervals from 2,000 patient-level bootstrap resamples (scans of one patient are resampled together;
  AbdomenAtlas 3.0 by scan).
- Head-to-head comparisons: paired bootstrap on the same resampled patients (2,000 resamples); a difference is
  significant when its 95% interval excludes zero; per-finding comparisons use Benjamini–Hochberg correction.
  Code: `scripts/paired_zs_bootstrap.py`, `scripts/ladder_stats.py --compare`.
- Encoders are scored on the full test sets; VLMs answer and write reports on the full CT-RATE and Merlin test sets
  (`--set full`, `scripts/vlm_full.py`) and answer on the 464 radiologist-annotated ReXGroundingCT scans.
