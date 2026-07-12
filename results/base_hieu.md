# model_size=base on divmix_hieu (vs small: real 2.72%, held-out 2.28%)
started Sun Jul 12 12:18:34 AM +07 2026, max_duration=500, LR 0.045, vocab 100
00:18 training done

## held-out (expanded 13165-word eval)
| epoch | held-out | khanh_toan | yen_nhi |
| ---: | ---: | ---: | ---: |
| 30 | 1.80% | 1.56% | 2.92% |
| 40 | 1.97% | 2.13% | 2.43% |
| 42 | 2.12% | 2.24% | 2.66% |
| 45 | 2.12% | 2.16% | 2.62% |
| 50 | 2.13% | 2.09% | 2.73% |
| 55 | 1.95% | 2.13% | 2.54% |
| 60 | 1.86% | 1.94% | 2.39% |

## per-speaker on 1000 real recordings (best epoch 30)
```
  Dung   1.98%
  Hieu   1.73%
  Khoi   2.07%
  Quan   1.08%
  Trung  3.64%
  ALL    1.84%
```
BASE_DONE
