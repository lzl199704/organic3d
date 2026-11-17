"""CT-RATE v2 test set = v1 validation minus the 37 brain series in dataset/metadata/no_chest_valid.txt (HF
ibrahimhamamci/CT-RATE, data_correction_note.md; copied to {work_root}/manifests/ctrate_v2/). Our CT-RATE test set
(3,005 after our exclusions) contains all 3,002 v2 volumes plus 3 of those brain series (valid_257_a_1/2/4), so every
CT-RATE metric is reported on the v2 set by dropping them."""
import os


def no_chest_volumes(work_root):
    p = os.path.join(work_root, 'manifests', 'ctrate_v2', 'dataset', 'metadata', 'no_chest_valid.txt')
    return {os.path.basename(l.strip()) for l in open(p) if l.strip()}
