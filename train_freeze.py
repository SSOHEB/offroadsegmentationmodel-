"""
EXP-5: ResNet34-UNet with FROZEN Encoder
========================================
Hypothesis: Freezing the pretrained encoder preserves ImageNet features
while allowing the decoder to learn domain-specific patterns.

Expected improvement: 0.55-0.65 IoU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Install segmentation-models-pytorch if not available
try:
    import segmentation_models_pytorch as smp
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'segmentation-models-pytorch'])
    import segmentation_models_pytorch as smp

# ========== REPRODUCIBILITY ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ========== CONFIGURATION ==========
DATASET_ROOT = "dataset_256"
NUM_CLASSES = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001  # Higher LR since only decoder trains
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
IMG_SIZE = (256, 256)

# ========== DATASET ==========
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.images = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_filename = self.images[idx].replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        if self.img_transform:
            image = self.img_transform(image)
        
        mask_np = np.array(mask)
        mask = torch.from_numpy(mask_np).long()
        
        if mask.ndim == 3:
            mask = mask.squeeze(0)
                
        return image, mask

# ========== TRAINING ==========
def calculate_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float().item()
        union = (pred_cls | target_cls).sum().float().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0

def main():
    print(f"🚀 EXP-5: ResNet34-UNet with FROZEN Encoder")
    print(f"Using Device: {DEVICE}")
    
    # Data transforms (NO ImageNet normalization - that hurt us before)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = SegmentationDataset(
        f"{DATASET_ROOT}/train/images",
        f"{DATASET_ROOT}/train/masks",
        img_transform=train_transform
    )
    val_dataset = SegmentationDataset(
        f"{DATASET_ROOT}/val/images",
        f"{DATASET_ROOT}/val/masks",
        img_transform=val_transform
    )
    
    NUM_WORKERS = 2
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0)
    )
    
    # ========== MODEL WITH FROZEN ENCODER ==========
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)
    
    # FREEZE the encoder!
    model.freeze_encoder()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Encoder FROZEN!")
    print(f"   Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Class weights
    class_counts = [
        59016700, 97779781, 311208907, 18074830, 72049226,
        45587751, 1263280, 19738472, 401180128, 619502525
    ]
    total_pixels = sum(class_counts)
    weights = [total_pixels / (NUM_CLASSES * c) for c in class_counts]
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    print(f"Using Class Weights: {class_weights}")
    
    # Loss and optimizer (only decoder parameters)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    best_val_iou = 0
    train_losses, val_losses = [], []
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_iou += calculate_iou(preds, masks, NUM_CLASSES)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, f"{CHECKPOINT_DIR}/last_model.pth")

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print(f"  💾 New best model saved! (IoU: {best_val_iou:.4f})")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('EXP-5: Frozen Encoder Training')
    plt.savefig(f"{CHECKPOINT_DIR}/training_curve.png")
    plt.close()
    
    print(f"\n✅ Training completed!")
    print(f"   Best Val IoU: {best_val_iou:.4f}")
    print(f"   Model saved to: {CHECKPOINT_DIR}/best_model.pth")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
