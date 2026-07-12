# BPE vocab-size experiment (new Trung, LR 0.045, reuse fbank_divmix_x8)
started: Sat Jul 11 09:25:53 AM +07 2026
baseline vocab=100 (LR winner): held-out 2.08% @ epoch ~42

=== Sat Jul 11 09:25:53 AM +07 2026 : VOCAB=250 -> ASR/zipformer/exp_bpe250_small_streaming_divmix_x8_newT_lr0045 ===

| epoch | held-out WER | khanh_toan | yen_nhi |
| ---: | ---: | ---: | ---: |
| 30 | EXPORT FAILED | | |
| 35 | EXPORT FAILED | | |
| 40 | EXPORT FAILED | | |
| 45 | EXPORT FAILED | | |
| 50 | EXPORT FAILED | | |
| 55 | EXPORT FAILED | | |
| 60 | EXPORT FAILED | | |

=== Sat Jul 11 11:54:01 AM +07 2026 : VOCAB=500 -> ASR/zipformer/exp_bpe500_small_streaming_divmix_x8_newT_lr0045 ===

| epoch | held-out WER | khanh_toan | yen_nhi |
| ---: | ---: | ---: | ---: |
| 30 | 14.88% | 10.38% | 19.38% |
| 35 | 7.79% | 6.57% | 9.00% |
| 40 | 6.92% | 7.27% | 6.57% |
| 45 | 6.23% | 0.00% | 12.46% |
| 50 | 11.76% | 7.27% | 16.26% |
| 55 | 5.88% | 3.81% | 7.96% |
| 60 | 2.77% | 0.00% | 5.54% |

=== Sat Jul 11 02:25:25 PM +07 2026 : ALL DONE ===
VOCAB_DONE

---

## FINAL VERDICT (on the expanded 13,165-word eval, LR 0.045, epoch 42)

| vocab | held-out WER | note |
| ---: | ---: | --- |
| 100 (current) | 1.57% | keep |
| 250 | 1.41% | TIE with 100 (bootstrap: -0.17pt, 95% CI [-0.48,+0.16], not significant) |
| 500 | 7.91% | decisively worse; unstable, yen_nhi collapses to 21.76% |

**Conclusion: keep vocab 100.** Bigger vocab does not help on this memorization task
(monosyllabic language, ~800 sentences, full sentence memorization leaves no room
for tokenization to matter). 500 hurts: too many token classes for the data, so
each is undertrained and the model destabilizes.

Note: the old 578-word eval showed 250=1.21% "beating" 100 (2.08%) -- that was
pure noise. The expanded 13,165-word eval (added 2026-07-11) settled it.
