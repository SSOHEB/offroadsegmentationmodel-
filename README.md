# Offroad Segmentation Project - HackDefence

This project implements a U-Net based semantic segmentation model for offroad terrain analysis.

## 📂 Project Structure

```
hackdefence/
├── train.py           # Training script (U-Net model training)
├── test.py            # Inference script (Generates predictions)
├── README.md          # Project documentation
├── checkpoints/       # Saved models (best_model.pth)
└── dataset_root/      # (Optional) Original dataset root
```

## 📊 Dataset Configuration

The project is currently configured to use the dataset at:
`C:/Users/ssohe/Desktop/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset`

**Expected Structure:**
```
Offroad_Segmentation_Training_Dataset/
├── train/
│   ├── Color_Images/   # Training images (.jpg)
│   └── Segmentation/   # Training masks (.png)
├── val/
│   ├── Color_Images/   # Validation images (.jpg)
│   └── Segmentation/   # Validation masks (.png)
└── test/               # (Missing in current dataset)
```

> **Note:** Since the `test` directory is missing in the current dataset, the `test.py` script has been configured to **fallback to the validation dataset** (`val/Color_Images`) for demonstration purposes.

## 🚀 How to Run

### 1. Training
To train the model:

```bash
python train.py
```

- **Configuration:** You can adjust `BATCH_SIZE`, `LEARNING_RATE`, and `NUM_EPOCHS` at the top of `train.py`.
- **Output:** The best model will be saved to `checkpoints/best_model.pth`.

### 2. Inference (Testing)
To generate segmentation masks using the trained model:

```bash
python test.py
```

- **Input:** Uses `test/Color_Images` if available, otherwise falls back to `val/Color_Images`.
- **Output:** Predictions are saved to `predictions/` (or `predictions_val_demo/` if using fallback).

## 🛠️ Recent Fixes & Changes

1.  **Dataset Paths:** Updated scripts to point to the correct absolute path on the Desktop.
2.  **Folder Names:** Corrected `train.py` to look for `Color_Images` and `Segmentation` folders instead of `images` and `masks`.
3.  **Test Fallback:** Added logic to `test.py` to handle the missing `test` folder by using validation images for inference.
4.  **Reproducibility:** Added seed setting (random, numpy, torch) to ensure consistent results.
5.  **Metric Removal in Test:** Removed all metric calculations (IoU/Accuracy) from `test.py` to strictly comply with inference-only requirements.
