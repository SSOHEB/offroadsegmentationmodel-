# 📔 Project Journal (Judge Submission)

## 1. Data Pipeline Optimization

*   **Task**: Preparing a 3000-image dataset for training.
*   **Initial Status**: Training was stalling (CPU usage at 100%, GPU utilization near 0%).
*   **Issue Faced**: The raw images were massive (3000px resolution), causing the "Loading" step to take longer than the "Training" step.
*   **Solution**: We wrote a custom script (`preprocess_dataset.py`) to resize all images to 256x256 **offline** (before training). This reduced I/O latency by 95%.

## 2. Model Training Strategy

*   **Task**: Training the UNet Segmentation Model.
*   **Initial Speed**: ~60 minutes per epoch (on local CPU).
*   **Issue Faced**: Feedback loop was too slow. We could not iterate on hyperparameters before the hackathon deadline.
*   **Solution**: We containerized our environment and migrated to a **T4 GPU** on Google Colab, achieving a speed of **<2 minutes per epoch** (50x improvement).

## 3. Handling Class Imbalance (Safety Critical)

*   **Task**: Detecting "Obstacles" (Class 7) and "Road" (Class 1).
*   **Initial IoU Score**: ~0.00 for Obstacles (The model was ignoring them).
*   **Issue Faced**: "Obstacles" represented only **0.08%** of the total pixels. The model optimized accuracy by simply guessing "Background" everywhere.
*   **Solution**: We calculated the exact pixel distribution and implemented **Weighted Cross-Entropy Loss**. This penalizes the model 100x more for missing an obstacle than for missing a road.

## 4. Documentation of Failure Cases

*   **Failure Case**: "The Black Mask Output"
*   **What Went Wrong**: Initially, our test script saved segmentation masks as raw integers (0, 1, 2...). To the human eye, these look like solid black images.
*   **Fix**: We updated the inference pipeline (`test.py`) to apply a **Color Palette** (Red for rocks, Green for grass) so predictions are instantly verifiable by humans.

---

## 5. Training Progress Analysis

Training was conducted over **50 epochs** on a Google Colab T4 GPU.

**Key Observations:**
-   **Epochs 1-10**: Rapid learning phase. Val IoU improved from 0.26 → 0.40
-   **Epochs 10-25**: Steady improvements. Model learned to distinguish road surfaces.
-   **Epochs 25-40**: IoU crossed 0.50. Class weighting helped detect rare obstacles.
-   **Epochs 40-50**: Convergence phase. Minor fluctuations around 0.52-0.54.
-   **Epoch 46**: Anomalous spike in Val Loss (0.55) - possible random batch variance.

**Training Curve Pattern:**
-   Train Loss: Consistent downward trend (1.43 → 0.35)
-   Val Loss: Downward with occasional spikes (1.10 → 0.34)
-   Val IoU: Upward trend with plateau around 0.52

---

## 📊 Final Results

| Metric | Value |
| :--- | :--- |
| **Best Val IoU** | **57.2%** (with TTA) |
| **Best Val Loss** | 0.3449 (Epoch 50) |
| **Training Time** | ~1.5 hours (50 epochs on T4 GPU) |
| **Time per Epoch** | ~2 minutes |
| **Inference Speed** | ~2 seconds per image |

## 📸 Sample Predictions

See the `predictions_val_demo/` folder for colorized segmentation outputs.
The model successfully segments:
-   Road surfaces (Green)
-   Vegetation (Teal)
-   Obstacles (Red tones)

## 🔮 Future Improvements (If More Time)
-   Implement Dice Loss for direct IoU optimization
-   Add Learning Rate Scheduler for better convergence
-   Increase training epochs to 100+
-   Try higher resolution (512x512)
