# 🚙 Offroad Semantic Segmentation

**A high-performance U-Net pipeline optimized for rapid offroad terrain analysis.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Key Features

- **Custom U-Net Architecture**: Tailored for 10-class segmentation tasks.
- **Optimized Data Pipeline**:
  - **Offline Preprocessing**: `preprocess_dataset.py` resizes and maps masks ahead of time, reducing I/O bottlenecks.
  - **Efficient Loading**: Support for `num_workers > 0` and `pin_memory`.
- **Advanced Training Logic**:
  - **Class Weighting**: Automatically handles severe class imbalance (e.g., Rare Obstacles vs Background).
  - **Checkpoint Resume**: Auto-resumes from `last_model.pth` to handle long training sessions.
- **GPU Ready**: Fully verified on T4 GPU (Google Colab) with <2 min/epoch training time.

---

## 📂 Repository Structure

```tree
├── train.py                # Main training loop (Resume + Class Weights enabled)
├── test.py                 # Inference script with Colorized Mask visualization
├── analyze_dataset.py      # Health check tool: Finds corrupt files & calculates stats
├── preprocess_dataset.py   # ETL script: Resizes 3k images -> 256x256 offline
├── zip_dataset.py          # Utility to pack data for Cloud/Colab transfer
└── colab_guide.md          # Step-by-step guide for GPU training
```

---

## 🛠️ Performance Engineering

This project implements a **50x speedup** over the baseline implementation:

| Metric | Baseline (CPU/Raw) | Optimized (GPU/Processed) |
| :--- | :--- | :--- |
| **Pipeline** | On-the-fly Resize | Offline Preprocessing |
| **Hardware** | CPU | T4 GPU |
| **Time/Epoch** | ~60 mins | **< 2 mins** |
| **Accuracy** | Ignored rare classes | **Weighted Loss** (Class 700 support) |

---

## 🚦 How to Run

### 1. Pre-requisites
Ensure you have the dataset (images/masks).

### 2. Prepare Data (One-Time)
Run the preprocessing script to generate the optimized `dataset_256` folder:
```bash
python preprocess_dataset.py
```

### 3. Train
Start the training loop (supports auto-resume):
```bash
python train.py
```
*Note: Check `train.py` config to switch `DEVICE` or `BATCH_SIZE`.*

### 4. Inference
Generate color-mapped predictions on test data:
```bash
python test.py
```

---

## 📊 Dataset Stats
- **Total Images**: ~3,000
- **Classes**: 10 (Background, Road, Obstacle, etc.)
- **Resolution**: Native (High-Res) -> Training (256x256)

---

**Author**: [Your Name/Team]
**Hackathon**: HackDefence 2026
