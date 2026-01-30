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

## 📝 Training Observations

| Epoch | Time | Train Loss | Val Loss | Val IoU | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | ~2m | 1.4268 | 1.1034 | 0.2586 | Initial run. Background learned. |
| **5** | | | | | |
| **10** | | | | | |
| **20** | | | | | |
| **30** | | | | | |
| **40** | | | | | |
| **50** | | | | | |

## 🧠 Hypotheses & Decisions

-   **Hypothesis**: Class weights will allow the model to detect 'Obstacle' (Class 7) despite it being only 0.08% of pixels.
-   **Decision**: Migrated to GPU because CPU training was projected at 40+ hours.
-   **Decision**: Used offline preprocessing because `test.py` showed significant I/O latency.

## 🔎 Post-Training Analysis Plan
1.  Check Class 7 (Obstacle) IoU specifically.
2.  Visualize prediction masks vs ground truth.
3.  Compare with baseline (if available).
