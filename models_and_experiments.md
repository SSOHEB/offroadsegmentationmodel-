# Models & Experiments Log

## Executive Summary
| Experiment | Model | Loss | Final IoU | Status |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-1** | Custom UNet | CE + Class Weights | **57.2%** (TTA) | ✅ **Best** |
| EXP-2 | Custom UNet | Dice+CE + Scheduler | 40.6% | ❌ Failed |
| EXP-3 | ResNet34-UNet | CE (ImageNet pretrained) | 45.0% | ❌ Lower |
| EXP-4 | Custom UNet | CE (100 epochs) | 51.1% | ⚠️ Oscillating |
| EXP-5 | ResNet34-UNet | CE (Frozen Encoder) | 42.5% | ❌ Lower |
| EXP-6 | DeepLabV3+ | CE | N/A | ❌ Crashed |
| EXP-7 | DeepLabV3+ | Focal+Dice Loss | N/A | ❌ Crashed |
| EXP-8 | EfficientNetV2-S | CE + Class Weights | 51.0% | ❌ Lower |
| EXP-9 | EfficientNetV2-FPN | CE + All Optimizations | 51.8% | ❌ Lower |


---

## Experiment 1: Baseline (BEST RESULT ✅)

### Configuration
| Parameter | Value |
| :--- | :--- |
| Model | Custom UNet (4 encoder, 4 decoder blocks) |
| Loss | `CrossEntropyLoss` with class weights |
| Optimizer | Adam (lr=0.001) |
| Epochs | 50 |
| Resolution | 256×256 |

### Results
-   **Final Val IoU**: **57.2%** (with TTA)
-   **Training Time**: ~1.5 hours
-   **Model File**: `checkpoints/best_model.pth`

---

## Experiment 2: Dice+CE Approach (FAILED ❌)

### Hypothesis
> "Combining Dice Loss with CrossEntropy will directly optimize IoU."

### Results
| Metric | EXP-1 | EXP-2 |
| :--- | :--- | :--- |
| **Final Val IoU** | **57.2%** (TTA) | 0.4057 |
| **Convergence** | Smooth | Slow, oscillating |

### Lesson Learned
> **Simpler is better.** CE with class weighting outperformed the Dice+CE combo.

---

## Experiment 3: ResNet34 Backbone (FAILED ❌)

### Hypothesis
> "Pretrained ResNet34 encoder will provide better features."

### Configuration
| Parameter | Value |
| :--- | :--- |
| Model | UNet with ResNet34 encoder (ImageNet) |
| Loss | CrossEntropyLoss (Weighted) |
| Optimizer | Adam (lr=0.0001) |
| Epochs | 50 |

### Results
-   **Final Val IoU**: 0.4500 (45%)
-   **Best Epoch**: 49 (0.4500)

### Why It Failed
1.  **ImageNet Normalization Mismatch**: Terrain images have different color distribution than natural images
2.  **Domain Gap**: ResNet learned cats/dogs, not terrain textures
3.  **Learning Rate Too Low**: 0.0001 was too conservative

---

## Experiment 4: UNet 100 Epochs (PARTIAL ⚠️)

### Hypothesis
> "More epochs will improve the baseline UNet beyond 0.52."

### Configuration
| Parameter | Value |
| :--- | :--- |
| Model | Custom UNet |
| Loss | CrossEntropyLoss (Weighted) |
| Epochs | **100** (vs 50 in baseline) |
| LR | 0.001 |

### Results
-   **Best Val IoU**: 0.5107 (Epoch 94)
-   **Final Val IoU**: 0.4968 (Epoch 100)
-   **Training Time**: ~5 hours

### Training Progression
| Epoch | Val IoU | Notes |
| :--- | :--- | :--- |
| 10 | 0.28 | Starting |
| 25 | 0.36 | Climbing |
| 50 | 0.39 | Below baseline |
| 75 | 0.48 | Approaching |
| 90 | 0.50 | Near baseline |
| 94 | **0.51** | Best |
| 100 | 0.50 | Final |

### Analysis
-   Heavy oscillation throughout training
-   LR (0.001) too aggressive for later epochs
-   Scheduler didn't reduce LR effectively
-   Did NOT beat baseline (0.52)

---

## Experiment 5: Frozen Encoder (COMPLETED ❌)

### Hypothesis
> "Freezing the pretrained ResNet encoder and only training the decoder will preserve ImageNet features while adapting to our domain."

### Results
-   **Final Val IoU**: 42.5%
-   **Issue**: Frozen features didn't adapt well to synthetic terrain

---

## Experiment 6: DeepLabV3+ (CRASHED ❌)

### Configuration
| Parameter | Value |
| :--- | :--- |
| Model | DeepLabV3+ (ResNet50 backbone) |
| Loss | CrossEntropyLoss |

### Results
-   **Status**: CRASHED — CUDA memory issues on Kaggle GPU

---

## Experiment 7: DeepLabV3+ with Focal+Dice (CRASHED ❌)

### Configuration
| Parameter | Value |
| :--- | :--- |
| Model | DeepLabV3+ |
| Loss | Focal + Dice combined |

### Results
-   **Status**: CRASHED — Same memory issues

---

## Experiment 8: EfficientNetV2-S (COMPLETED ❌)

### Hypothesis
> "EfficientNetV2 is state-of-the-art, should outperform custom UNet."

### Configuration
| Parameter | Value |
| :--- | :--- |
| Model | EfficientNetV2-S encoder + UNet decoder |
| Loss | CrossEntropyLoss (Weighted) |
| Epochs | 30 |

### Results
-   **Final Val IoU**: 51.0%
-   **Analysis**: Pretrained features didn't help with synthetic data

---

## Experiment 9: EfficientNetV2-FPN (COMPLETED ❌)

### Hypothesis
> "FPN decoder with all optimizations will beat baseline."

### Configuration
| Parameter | Value |
| :--- | :--- |
| Model | EfficientNetV2-S + FPN decoder |
| Loss | CrossEntropyLoss (Weighted) |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingWarmRestarts |
| Epochs | 30 |

### Results
-   **Final Val IoU**: 51.8%
-   **Best at Epoch**: 29

### Key Insight
> **Simple Custom UNet (57.2%) beats complex EfficientNetV2-FPN (51.8%)**
> Pretrained ImageNet models don't help with synthetic terrain data.

---

## Techniques Summary

### ✅ Worked
| Technique | Impact |
| :--- | :--- |
| **Class Weighting** | High — rare classes detectable |
| **Offline Preprocessing** | High — 95% I/O reduction |
| **GPU Migration** | High — 34× speedup |
| **Test-Time Augmentation (TTA)** | +2% IoU boost |

### ❌ Didn't Work
| Technique | Issue |
| :--- | :--- |
| **Dice+CE Loss** | Slower convergence |
| **Pretrained ResNet/EfficientNet** | Domain gap hurt accuracy |
| **100 epochs** | Oscillation, no improvement |
| **DeepLabV3+** | Memory crashes |

---

## Files Produced

| File | Experiment | IoU |
| :--- | :--- | :--- |
| `best_model.pth` | EXP-1 | **57.2%** (TTA) |
| `resnet_checkpoints/` | EXP-3 | 45.0% |
| `unet_100epoch_checkpoints/` | EXP-4 | 51.1% |
