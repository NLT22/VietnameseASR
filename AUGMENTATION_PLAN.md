# VietnameseASR augmentation plan

## What the current results mean

The `vi_asr_corpus` result below 10% WER is not a useful generalization
baseline. It contains only three unique transcripts, and all dev/test
transcripts also occur in train.

`VietnameseASR` is substantially harder and more realistic:

- 584 original training recordings and 584 unique training transcripts
- 73 dev and 73 test recordings with no exact transcript overlap with train
- about 51 words per utterance
- best existing test WER: 74.87%
- the old 20-second feature filter omitted 9 dev and 15 test utterances
- `run_x10.sh` creates ten noisy copies of every training utterance, but does
  not add any linguistic diversity

The next goal should be a reproducible improvement over 74.87% WER. Reaching
below 10% on unseen long sentences with only about 2.4 hours of original
training speech is not a realistic immediate target.

## Recommended first experiment

Use the original recordings with:

1. 0.9x and 1.1x speed perturbation.
2. Dynamic MUSAN mixing at training time.
3. SpecAugment at training time.
4. All utterances up to 40 seconds.
5. A smaller Zipformer, CTC auxiliary loss, and checkpoint averaging.

This is implemented in `run_robust.sh`.

```bash
conda activate /media/pc/fa8cd839-121b-4df4-9c0c-0958d9216d28/icefall/test-icefall
cd /media/pc/fa8cd839-121b-4df4-9c0c-0958d9216d28/icefall/egs/VietnameseASR

python3 local/audit_corpus.py
bash run_robust.sh --stage 0 --stop_stage 3
```

Stage 0 rebuilds clean splits from `dataset/`, so it intentionally replaces
the generated x10 audio and transcript rows. The original `dataset/` is not
modified.

## Experiment order

Run controlled ablations instead of changing everything repeatedly:

1. Clean originals + speed perturbation + SpecAugment.
2. Add online MUSAN mixing.
3. Compare `avg=5`, `avg=10`, and `avg=15`.
4. If Zipformer remains weak, fine-tune a multilingual self-supervised model
   such as XLS-R. For this amount of labeled speech, pretrained multilingual
   representations are likely more valuable than increasingly aggressive
   waveform augmentation.

Avoid treating permanent noisy copies as new information. They can improve
noise robustness, but they repeat the same labels and cannot solve unseen-word
or unseen-sentence generalization.

## References

- SpecAugment: https://arxiv.org/abs/1904.08779
- Speed perturbation: https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf
- MUSAN: https://arxiv.org/abs/1510.08484
- XLS-R: https://arxiv.org/abs/2111.09296
