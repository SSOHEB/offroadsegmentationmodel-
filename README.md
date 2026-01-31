# 🚙 Offroad Semantic Segmentation

**AI-powered terrain analysis for autonomous offroad navigation**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![IoU](https://img.shields.io/badge/Val%20IoU-54.9%25-brightgreen)

**Hackathon**: Duality AI - Offroad Autonomy Segmentation Challenge  
**Final IoU Score**: **54.9%**

---

## 📁 Submission Contents

```
├── train.py               # Training script
├── test.py                # Inference script (with TTA)
├── checkpoints/
│   └── best_model.pth     # Trained model weights
├── HACKATHON_REPORT.md    # Detailed 8-page report
├── models_and_experiments.md  # Experiment log
└── README.md              # This file
```

---

## 🚀 Quick Start (Reproduction Steps)

### 1. Environment Setup

```bash
# Option A: Using conda (recommended)
conda create -n offroad python=3.8
conda activate offroad
pip install torch torchvision tqdm pillow numpy

# Option B: Using the EDU environment (from Duality)
conda activate EDU
```

### 2. Dataset Preparation

Place the dataset in the following structure:
```
dataset_256/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    └── images/   # testImages from Duality
```

**If using raw dataset**, run preprocessing first:
```bash
python preprocess_dataset.py
```

### 3. Training (Optional - model already trained)

```bash
python train.py
```

**Expected output**:
- Training logs printed to console
- Checkpoints saved to `checkpoints/`
- Best model: `checkpoints/best_model.pth`

### 4. Inference on Test Images

```bash
python test.py
```

**Expected output**:
- Colored segmentation masks saved to `predictions/`
- Each output mask is RGB-colored for visualization

---

## 🏆 Results

| Metric | Value |
| :--- | :--- |
| **Validation IoU** | **52.6%** |
| **Model Architecture** | Custom UNet |
| **Training Time** | 1.5 hours (T4 GPU) |
| **Inference Speed** | ~50 FPS |

---

## 🔧 Configuration

Edit `train.py` to modify:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `NUM_EPOCHS` | 50 | Training epochs |
| `BATCH_SIZE` | 8 | Batch size |
| `LEARNING_RATE` | 0.001 | Adam learning rate |
| `IMG_SIZE` | (256, 256) | Input resolution |

---

## 📊 Class Mapping

| ID | Class Name | Color |
| :--- | :--- | :--- |
| 100 | Trees | Dark Red |
| 200 | Lush Bushes | Green |
| 300 | Dry Grass | Yellow |
| 500 | Dry Bushes | Blue |
| 550 | Ground Clutter | Purple |
| 600 | Flowers | Cyan |
| 700 | Logs | Gray |
| 800 | Rocks | Dark Gray |
| 7100 | Landscape | Dark Red |
| 10000 | Sky | Red |

---

## 📈 Key Techniques

1. **Class Weighting**: Inverse-frequency weighting handles severe class imbalance
2. **Test-Time Augmentation**: +2% IoU boost with horizontal/vertical flip averaging
3. **Offline Preprocessing**: 95% I/O reduction for faster training

---

## 📝 Output Interpretation

**Prediction masks** are saved as RGB images where each color represents a class:
- View predictions in the `predictions/` folder
- Compare with ground truth masks visually

---

## 📖 Documentation

- `HACKATHON_REPORT.md` - Full methodology and analysis
- `models_and_experiments.md` - All experiments conducted

---

**Author**: S SOHEB  
**Contact**: [GitHub](https://github.com/SSOHEB)
