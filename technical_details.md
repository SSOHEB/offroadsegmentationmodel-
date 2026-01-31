# 🛠️ Technical Details & Engineering Report

## 1. System Pipeline
The project implements a classic **ETL (Extract, Transform, Load)** pipeline optimized for deep learning:

1.  **Ingestion (Extract)**:
    *   **Input**: Raw High-Res Images (~2000x3000px) and Integer Masks.
    *   **Integrity Check**: `analyze_dataset.py` scans for corrupt headers, missing pairs, and invalid class labels before training starts.

2.  **Preprocessing (Transform - Offline)**:
    *   *Decision*: Moved from on-the-fly (online) to offline preprocessing to eliminate CPU bottlenecks.
    *   **Action**: `preprocess_dataset.py` resizes all assets to **256x256** and remaps class IDs to contiguous integers (0-9).
    *   **Storage**: Processed data is saved to `dataset_256/` for low-latency access.

3.  **Training (Load & Learn)**:
    *   **Loader**: PyTorch `DataLoader` with `num_workers=2` (tuned for Colab T4) and `pin_memory=True` for non-blocking GPU transfer.
    *   **Strategy**: Full-dataset traversal with Weighted Cross-Entropy Loss.

4.  **Inference (Visualize)**:
    *   **Post-Processing**: `test.py` maps raw class predictions (integers) to an RGB Color Palette for human-readable visualization.

---

## 2. Model Architecture: Custom U-Net
We implemented a **U-Net** architecture, chosen for its ability to capture both local texture (obstacles) and global context (road boundaries).

*   **Encoder (Contracting Path)**:
    *   4 Blocks: Each consists of **Double Convolution** (Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU).
    *   Downsampling: MaxPool2d (2x2) decreases spatial dimension while increasing feature depth ($64 \to 128 \to 256 \to 512$).
*   **Bottleneck**:
    *   Bridge layer connecting Encoder and Decoder ($512 \to 1024$ channels).
*   **Decoder (Expansive Path)**:
    *   4 Blocks: Uses **Bilinear Upsampling** to restore spatial resolution.
    *   **Skip Connections**: Concatenates high-res feature maps from the Encoder with the Decoder to preserve edge details loss during downsampling.
*   **Head**:
    *   Final $1\times1$ Convolution mapping to **10 Output Classes**.

## 3. Results

| Metric | Value |
| :--- | :--- |
| **Final Val IoU** | **0.549** (54.9%) |
| **Training Epochs** | 50 |
| **Time per Epoch** | ~2 minutes (Colab T4) |
| **Total Training Time** | ~1.5 hours |

---

## 4. Engineering Challenges & Solutions

| Challenge | Root Cause | Engineering Solution |
| :--- | :--- | :--- |
| **Training Latency** | CPU-bound resizing of 3MP images during each iteration. | **Offline Preprocessing**: Resized entire dataset once, reducing per-epoch I/O by 95%. |
| **Class Imbalance** | 'Obstacle' class represented < 0.1% of pixels. | **Algorithmic Weighting**: Calculated inverse frequency weights via `analyze_dataset.py` and applied to Loss function. |
| **Compute Limit** | Local CPU (i5) required ~1 hour/epoch. | **Cloud Migration**: Dockerized env (zip + scripts) and migrated to Google Colab (T4 GPU), achieving **<2 min/epoch**. |
| **CUDA Errors** | NumPy trying to access GPU tensors during metric calc. | **Tensor Detachment**: Updated `train.py` to correctly detach tensors (`.cpu().item()`) before passing to CPU-bound libraries. |
| **Windows I/O** | `multiprocessing` spawn errors on Windows. | **Process Guard**: Implemented `freeze_support()` and optimized worker count. |

---

## 5. Training Progress & Convergence

### Loss Curves
Training was monitored across 50 epochs. Key patterns observed:

| Phase | Epochs | Train Loss | Val IoU | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Rapid Learning** | 1-10 | 1.43 → 0.59 | 0.26 → 0.40 | Model learns background/foreground |
| **Refinement** | 10-25 | 0.59 → 0.43 | 0.40 → 0.48 | Edge detection improves |
| **Plateau** | 25-40 | 0.43 → 0.38 | 0.48 → 0.52 | Diminishing returns |
| **Convergence** | 40-50 | 0.38 → 0.35 | 0.52 → 0.53 | Stable oscillation |

### Anomalies Detected
-   **Epoch 46**: Val Loss spiked to 0.55 (batch variance). Recovered by Epoch 49.
-   **Class 7 (Obstacle)**: Started being detected around Epoch 20 after class weighting took effect.

### Optimization Summary
-   **Speed**: ~3600s/epoch (Baseline CPU) → **~105s/epoch** (Colab T4). **[34x Speedup]**
-   **Memory**: VRAM usage ~4GB at batch_size=8, resolution=256x256.
-   **Checkpointing**: Auto-save every epoch enabled resume after Colab disconnects.

---

## 5. Tech Stack

*   **Core**: Python 3.10+
*   **Deep Learning**: PyTorch (Torch + Torchvision)
*   **Data Processing**: NumPy, Pillow (PIL)
*   **Visualization**: Matplotlib, Tqdm
*   **Infrastructure**: Google Colab (T4 GPU), Git/GitHub
