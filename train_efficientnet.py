import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Install dependencies if needed
try:
    import segmentation_models_pytorch as smp
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'segmentation-models-pytorch'])
    import segmentation_models_pytorch as smp

try:
    import timm
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'timm'])
    import timm

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
# KAGGLE PATH - Update if different
DATASET_ROOT = "/kaggle/input/dataset-256"
NUM_CLASSES = 10
BATCH_SIZE = 8  # EfficientNetV2-S uses more memory, reduce if OOM
LEARNING_RATE = 0.0001  # Lower LR for pretrained
NUM_EPOCHS = 5  # Quick test run
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "/kaggle/working/checkpoints_efficientnet"
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

# ========== DICE LOSS ==========
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0,3,1,2).float()
        intersection = (pred * target_one_hot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight, ce_ratio=0.5, dice_ratio=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice = DiceLoss()
        self.ce_ratio = ce_ratio
        self.dice_ratio = dice_ratio
        
    def forward(self, pred, target):
        return self.ce_ratio * self.ce(pred, target) + self.dice_ratio * self.dice(pred, target)

# ========== METRICS ==========
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
    print(f"🚀 EfficientNetV2-S UNet Training (5 Epoch Test)")
    print(f"Using Device: {DEVICE}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    
    print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    NUM_WORKERS = 2
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0),
        drop_last=True  # Prevents BatchNorm error when last batch has only 1 sample
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0),
        drop_last=True  # Prevents BatchNorm error
    )
    
    # ========== EFFICIENTNETV2-S UNET MODEL ==========
    print("📦 Loading EfficientNetV2-S encoder with ImageNet weights...")
    model = smp.Unet(
        encoder_name="timm-efficientnetv2_rw_s",  # EfficientNetV2-S
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model loaded: {total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable)")
    
    # Class weights
    class_counts = [
        59016700, 97779781, 311208907, 18074830, 72049226,
        45587751, 1263280, 19738472, 401180128, 619502525
    ]
    total_pixels = sum(class_counts)
    weights = [total_pixels / (NUM_CLASSES * c) for c in class_counts]
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    
    # Loss: Combined Dice + CE for better IoU
    criterion = CombinedLoss(ce_weight=class_weights, ce_ratio=0.5, dice_ratio=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_iou = 0
    train_losses, val_losses, val_ious = [], [], []
    
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
        val_ious.append(avg_val_iou)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Highlight if IoU > 0.6
        iou_indicator = "🎯" if avg_val_iou >= 0.6 else ""
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f} {iou_indicator}, LR: {current_lr:.6f}")
        
        # Save best model by IoU
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print(f"  💾 New best model saved! IoU: {best_val_iou:.4f}")
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"📊 EFFICIENTNETV2-S TEST RESULTS (5 Epochs)")
    print(f"{'='*50}")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Final Val IoU: {val_ious[-1]:.4f}")
    print(f"Target (0.6+): {'✅ ACHIEVED!' if best_val_iou >= 0.6 else '❌ Not yet'}")
    print(f"{'='*50}")
    
    if best_val_iou >= 0.5:
        print(f"💡 Promising! With more epochs, 0.6+ IoU is likely achievable.")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')
    
    ax2.plot(val_ious, label='Val IoU', color='green', marker='o')
    ax2.axhline(y=0.6, color='r', linestyle='--', label='Target (0.6)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.set_title('Validation IoU')
    
    plt.tight_layout()
    plt.savefig(f"{CHECKPOINT_DIR}/training_curves.png")
    plt.close()
    
    print(f"📈 Training curves saved to {CHECKPOINT_DIR}/training_curves.png")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
