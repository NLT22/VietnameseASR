#!/usr/bin/env bash
# Fires when the GPU frees (vocab experiment done). Regenerates the held-out eval
# at 100 sentences x 5 voices (~5000 words, was 578), then re-baselines the
# deployed model on it. Detached/reset-proof. -> results/heldout_expanded.md
set -u
cd /media/pc/c88ba509-53f0-4c97-9e44-e33483754b08/icefall/egs/VietnameseASR
GWEN=/media/pc/c88ba509-53f0-4c97-9e44-e33483754b08/gwen-tts/.venv-gwen/bin/python
RES=results/heldout_expanded.md; mkdir -p results
echo "# Expanded held-out eval (100 sentences x 5 voices)" > "$RES"
echo "waiting for GPU (vocab experiment to finish)..." >> "$RES"
# wait for vocab experiment done AND GPU free
while pgrep -f run_vocab_experiment.sh >/dev/null 2>&1; do sleep 120; done
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 60; done
echo "$(date +%H:%M:%S) GPU free, generating..." >> "$RES"

# backup old set, regenerate fresh
[ -d heldout_speaker_eval ] && rm -rf heldout_speaker_eval_25x2_backup && mv heldout_speaker_eval heldout_speaker_eval_25x2_backup
$GWEN local/tts/make_heldout_speaker_eval.py >> "$RES" 2>&1
n=$(tail -n +2 heldout_speaker_eval/manifest.tsv 2>/dev/null | wc -l)
echo "$(date +%H:%M:%S) generated $n clips" >> "$RES"

# re-baseline the deployed model on the expanded set (per-voice)
echo "" >> "$RES"; echo "## Deployed model on expanded eval" >> "$RES"; echo '```' >> "$RES"
$GWEN eval_heldout_speaker.py --model-dir deploy/jetson_nano/model_divmix_x8_newT_lr0045_epoch42_avg10 >> "$RES" 2>&1
echo '```' >> "$RES"
echo "EXPAND_DONE" >> "$RES"
