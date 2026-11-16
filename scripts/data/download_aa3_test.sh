#!/bin/bash
# AbdomenAtlas 3.0 (RadGPT, Bassi et al., ICCV 2025; Hugging Face AbdomenAtlas/AbdomenAtlas3.0Mini, CC BY-NC-SA 4.0)
# IID test split only (926 scans, the paper's 10% test set; TrainTestIDS/IID_test.csv). The release ships images and
# masks as 232-case tar archives; each archive is streamed and only the test cases are written (no archive is stored).
# Output: $AA3_ROOT/IID_test/BDMAP_xxxxxxxx/{ct.nii.gz, segmentations/*.nii.gz}. Resumable (finished archives are
# listed in $AA3_ROOT/logs/done.txt). Needs a Hugging Face token with access to the dataset in $HF_TOKEN.
#
#   AA3_ROOT=/path/to/AbdomenAtlas3.0 HF_TOKEN=hf_... bash scripts/data/download_aa3_test.sh
# then: AA3_ROOT=... python scripts/data/prepare_aa3.py --stage resample / ts / finalize
set -u
D=${AA3_ROOT:?set AA3_ROOT}
T=${HF_TOKEN:?set HF_TOKEN}
R=https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini/resolve/main
mkdir -p "$D/IID_test" "$D/logs" "$D/TrainTestIDS"
for f in AbdomenAtlas3.0MiniWithMeta.csv TrainTestIDS/IID_test.csv; do
  [ -f "$D/$f" ] || curl -sfL -H "Authorization: Bearer $T" "$R/$f" -o "$D/$f"
done
tail -n +2 "$D/TrainTestIDS/IID_test.csv" | tr -d '\r' | sed 's#$#/*#' > "$D/IID_test_patterns.txt"
# archive names: 40 image + 40 mask archives of 232 cases each (BDMAP_00000001 ... BDMAP_00009262)
python - "$D" <<'PY' > "$D/tar_list.txt"
import sys
for kind, pre in (('mask_only', 'masks'), ('image_only', 'images')):
    for s in range(1, 9263, 232):
        e = min(s + 231, 9262)
        print(f'{kind}/AbdomenAtlas3_{pre}_BDMAP_BDMAP_{s:08d}_BDMAP_{e:08d}.tar.gz')
PY
one() {
  f=$1
  grep -qxF "$f" "$D/logs/done.txt" 2>/dev/null && return
  for a in 1 2 3; do
    curl -sfL --retry 5 -H "Authorization: Bearer $T" "$R/$f" \
      | tar -xz -C "$D/IID_test" --wildcards -T "$D/IID_test_patterns.txt" 2>>"$D/logs/tar_err.log"
    st=("${PIPESTATUS[@]}")
    if [ "${st[0]}" = 0 ]; then echo "$f" >> "$D/logs/done.txt"; echo "$(date +%T) done $f"; return; fi
    echo "$(date +%T) retry $a $f (curl ${st[0]} tar ${st[1]})"
  done
}
export -f one; export D T R
xargs -P 4 -I{} bash -c 'one {}' < "$D/tar_list.txt"
echo "$(date +%T) $(ls "$D/IID_test" | wc -l) test cases"
