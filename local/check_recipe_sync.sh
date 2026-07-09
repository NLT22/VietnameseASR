#!/usr/bin/env bash
# Fail if shared code drifts between vi_asr_corpus (canonical repo) and VietnameseASR.
# Same recipe on different datasets; only data/exp_* output dirs may differ.
# Lives in the vi_asr_corpus repo so it is version-tracked. Degrades to a no-op
# if the VietnameseASR sibling dir is absent (e.g. a fresh standalone clone).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # egs/vi_asr_corpus/local
a="$(dirname "$here")"                                  # egs/vi_asr_corpus
b="$(dirname "$a")/VietnameseASR"                       # egs/VietnameseASR

# ponytail: explicit file list, not a glob walk — exp_*/data dirs legitimately differ.
files=(
  run.sh run_x10.sh
  local/prepare_manifests.py local/prepare_vi_asr_corpus.py
  local/augment_train_with_musan.py local/audit_dataset.py
  local/prepare_matched_splits.py local/export_for_jetson.sh
  local/compute_fbank.py local/train_bpe_model.py local/prepare_lang_bpe.py
  ASR/zipformer/train.py ASR/zipformer/decode.py ASR/zipformer/ctc_decode.py
  ASR/zipformer/export.py ASR/zipformer/streaming_decode.py
)

if [ ! -d "$b" ]; then
  echo "recipe sync OK (VietnameseASR sibling absent, nothing to compare)"
  exit 0
fi

drift=0
for f in "${files[@]}"; do
  [ -f "$a/$f" ] && [ -f "$b/$f" ] || continue
  if ! diff -q "$a/$f" "$b/$f" >/dev/null; then
    echo "DRIFT: $f differs between vi_asr_corpus and VietnameseASR" >&2
    drift=1
  fi
done

if [ "$drift" = 1 ]; then
  echo "Recipes must share identical code. Sync the file(s) above, then commit." >&2
  exit 1
fi
echo "recipe sync OK"
