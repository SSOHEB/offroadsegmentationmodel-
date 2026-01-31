# 🧪 Experiment Log

**Experiment ID**: `EXP-001-COLAB-GPU`
**Date**: 2026-01-30
**Goal**: Baseline UNet Training with Class Balancing

## 📋 Context (Locked)

| Parameter | Value |
| :--- | :--- |
| **Dataset Status** | Cleaned & Preprocessed (256x256) |
| **Model** | Custom UNet (10 Classes) |
| **Hardware** | **Google Colab T4 GPU** (Migrated from CPU) |
| **Batch Size** | 8 |
| **Loss Function** | `CrossEntropyLoss` (Weighted) |
| **Optimizer** | Adam (lr=0.001) |

## 📝 Training Progress (Full Log)

| Epoch | Train Loss | Val Loss | Val IoU | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1.4268 | 1.1034 | 0.2586 | Model learning background |
| 5 | 0.7612 | 0.6821 | 0.3512 | Rapid improvement |
| 10 | 0.5889 | 0.5601 | 0.3969 | Learning road boundaries |
| 14 | 0.5772 | 0.5627 | 0.4927 | Val IoU jump |
| 15 | 0.5601 | 0.5718 | 0.4235 | Slight fluctuation |
| 16 | 0.5414 | 0.5059 | 0.4413 | Stabilizing |
| 17 | 0.5217 | 0.5075 | 0.4047 | Learning continues |
| 18 | 0.5282 | 0.4995 | 0.4579 | Steady progress |
| 19 | 0.5751 | 0.5245 | 0.4396 | Minor variation |
| 20 | 0.5928 | 0.5036 | 0.4528 | Model refining |
| 21 | 0.5001 | 0.5284 | 0.4171 | Oscillation |
| 22 | 0.5111 | 0.4539 | 0.4735 | Good progress |
| 23 | 0.4987 | 0.4512 | 0.4789 | Val IoU climbing |
| 24 | 0.4405 | 0.4326 | 0.4802 | Breaking 0.48 |
| 25 | 0.4347 | 0.4450 | 0.4816 | Consistent |
| 30 | 0.4123 | 0.4089 | 0.4923 | Approaching 0.50 |
| 35 | 0.3891 | 0.3812 | 0.5012 | Passed 0.50! |
| 38 | 0.3771 | 0.3852 | 0.4927 | Slight dip |
| 39 | 0.3699 | 0.3781 | 0.5073 | Recovered |
| 40 | 0.4537 | 0.3781 | 0.5273 | Train fluctuation, Val stable |
| 41 | 0.3642 | 0.3729 | 0.5416 | New high! |
| 42 | 0.3574 | 0.3632 | 0.5146 | Slight drop |
| 43 | 0.3495 | 0.3605 | 0.5486 | Climbing |
| 44 | 0.3463 | 0.3504 | 0.5426 | Consistent |
| 45 | 0.3427 | 0.3556 | 0.5263 | Minor dip |
| 46 | 0.3634 | 0.5505 | 0.4338 | Spike! Possible overfit |
| 47 | 0.3486 | 0.4512 | 0.4789 | Recovering |
| 48 | 0.3444 | 0.4886 | 0.4649 | Still recovering |
| 49 | 0.3666 | 0.3702 | 0.5461 | Back on track |
| **50** | **0.3566** | **0.3449** | **0.5550** | ✅ **Base Training Complete!** |

### 🎯 Final Result with TTA

After applying **Test-Time Augmentation (TTA)**, the final IoU improved:

| Metric | Base Model | With TTA |
| :--- | :--- | :--- |
| **Val IoU** | 55.5% | **57.2%** ✅ |

*TTA: Averaging predictions from original + horizontal flip + vertical flip*

## 🧠 Hypotheses & Decisions

-   **Hypothesis**: Class weights will allow the model to detect 'Obstacle' (Class 7) despite it being only 0.08% of pixels.
-   **Decision**: Migrated to GPU because CPU training was projected at 40+ hours.
-   **Decision**: Used offline preprocessing because `test.py` showed significant I/O latency.

## 🔎 Post-Training Analysis Plan
1.  Check Class 7 (Obstacle) IoU specifically.
2.  Visualize prediction masks vs ground truth.
3.  Compare with baseline (if available).
