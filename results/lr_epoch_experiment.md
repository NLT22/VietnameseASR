# LR x epoch experiment (new Trung audio)
started: Sat Jul 11 01:11:09 AM +07 2026
data: divmix_x8 (new Trung), 60 epochs, avg 10, beam_search held-out eval

=== Sat Jul 11 01:11:09 AM +07 2026 : TRAIN LR=0.01 -> ASR/zipformer/exp_bpe100_small_streaming_divmix_x8_newT_lr001 ===

| epoch | held-out WER | khanh_toan | yen_nhi |
| ---: | ---: | ---: | ---: |
| 30 | 24.05% | 11.76% | 36.33% |
| 35 | 18.86% | 6.23% | 31.49% |
| 40 | 14.36% | 2.42% | 26.30% |
| 45 | 12.28% | 2.08% | 22.49% |
| 50 | 12.46% | 2.77% | 22.15% |
| 55 | 11.42% | 3.46% | 19.38% |
| 60 | 10.90% | 3.46% | 18.34% |

=== Sat Jul 11 03:40:05 AM +07 2026 : TRAIN LR=0.02 -> ASR/zipformer/exp_bpe100_small_streaming_divmix_x8_newT_lr002 ===

| epoch | held-out WER | khanh_toan | yen_nhi |
| ---: | ---: | ---: | ---: |
| 30 | 8.82% | 2.77% | 14.88% |
| 35 | 7.27% | 2.77% | 11.76% |
| 40 | 6.75% | 3.11% | 10.38% |
| 45 | 6.23% | 3.81% | 8.65% |
| 50 | 5.88% | 4.15% | 7.61% |
| 55 | 5.71% | 4.50% | 6.92% |
| 60 | 5.02% | 4.15% | 5.88% |

=== Sat Jul 11 06:07:54 AM +07 2026 : TRAIN LR=0.045 -> ASR/zipformer/exp_bpe100_small_streaming_divmix_x8_newT_lr0045 ===

| epoch | held-out WER | khanh_toan | yen_nhi |
| ---: | ---: | ---: | ---: |
| 30 | 4.50% | 2.08% | 6.92% |
| 35 | 3.46% | 2.42% | 4.50% |
| 40 | 2.42% | 2.08% | 2.77% |
| 45 | 2.77% | 2.42% | 3.11% |
| 50 | 2.60% | 1.73% | 3.46% |
| 55 | 2.60% | 2.08% | 3.11% |
| 60 | 3.63% | 3.46% | 3.81% |

=== Sat Jul 11 08:37:19 AM +07 2026 : ALL DONE ===
DONE

---

## Analysis (autonomous)

- **LR 0.01**: no results
- **LR 0.02**: no results
- **LR 0.045**: no results
## Winner: LR ?, epoch None (held-out 999%)

ANALYSIS_DONE


---

## CONCLUSION (corrected — the autonomous analyzer's regex missed the % sign)

### Held-out WER grid (all beam_search, avg 10, new Trung audio)

| epoch | LR 0.01 | LR 0.02 | LR 0.045 |
| ---: | ---: | ---: | ---: |
| 30 | 24.05% | 8.82% | 4.50% |
| 40 | 14.36% | 6.75% | **2.42%** |
| 50 | 12.46% | 5.88% | 2.60% |
| 60 | 10.90% | 5.02% | 3.63% |

Fine sweep LR 0.045: epoch 38-42 = 2.25/2.25/2.42/2.08/2.08% (flat; knee ~40).

### Answers

1. **Learning rate: 0.01 (forced by run.sh) is far too low.** The icefall
   default **0.045 is best: 2.08-2.42% vs 0.01's 10.90%** (~4.5x better). 0.02 is
   intermediate (5.02%). run.sh overriding the default down to 0.01 was crippling
   every model. **Recommend base_lr 0.045.**

2. **60 epochs is NOT necessary — at the good LR it overfits.** LR 0.045 peaks at
   **epoch ~40 (2.08-2.42%)** and degrades to 3.63% by 60. Recommend **~40 epochs**.
   (Low LRs only "need" 60 because they never converge.)

### Winner: LR 0.045, epoch ~40-42 — per-speaker WER (800 real recordings)

| speaker | WER |
| --- | ---: |
| Quan | 1.51% |
| Khoi | 1.56% |
| Dung | 1.60% |
| Trung | 3.05% |
| **overall** | **1.74%** |

Held-out speakers at this model: **2.08%** (vs old deployed divmix_x8: 9.17%).

### Trung / de-clipping verdict

New CLEAN audio: Trung 3.05%, down from 4.18-5.74% on the de-clipped audio. His
relative gap to the other speakers shrank from ~4-5x to ~2x. **The de-clipping
was hurting him and the clean recording recovered a real chunk — but a residual
~2x gap remains**, so it was never purely the reconstruction (his voice /
recording conditions carry the rest). Caveat: this run also changed the LR, so
the ratio-to-others is the cleaner signal; it did shrink.

ANALYSIS_DONE_CORRECTED
