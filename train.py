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
DATASET_ROOT = "dataset_256" # UPDATED: Use preprocessed data
NUM_CLASSES = 10  # Mapped from raw values: 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000
BATCH_SIZE = 8 # UPDATED: Increased for faster training with smaller images
LEARNING_RATE = 0.001
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
        # mask_transform removed (not needed for preprocessed data)
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
        
        # Mask is already 0-9 and 256x256. Just convert to LongTensor.
        mask_np = np.array(mask)
        mask = torch.from_numpy(mask_np).long()
        
        if mask.ndim == 3:
            mask = mask.squeeze(0)
                
        return image, mask

# ========== UNET MODEL ==========
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Decoder
        self.dec3 = DoubleConv(256 + 512, 256)
        self.dec2 = DoubleConv(128 + 256, 128)
        self.dec1 = DoubleConv(64 + 128, 64)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        
        return self.final(d1)

# ========== DICE LOSS (New for IoU Optimization) ==========
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
    """Combines CrossEntropy (pixel accuracy) with Dice (IoU optimization)"""
    def __init__(self, ce_weight, ce_ratio=0.5, dice_ratio=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice = DiceLoss()
        self.ce_ratio = ce_ratio
        self.dice_ratio = dice_ratio
        
    def forward(self, pred, target):
        return self.ce_ratio * self.ce(pred, target) + self.dice_ratio * self.dice(pred, target)

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
    # ========== DATA AUGMENTATION (New) ==========
    # Train transform: Augment for diversity
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
    ])
    
    # Validation transform: No augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets (Updated paths for dataset_256 structure)
    train_dataset = SegmentationDataset(
        f"{DATASET_ROOT}/train/images",
        f"{DATASET_ROOT}/train/masks",
        img_transform=train_transform  # WITH Augmentation
    )
    val_dataset = SegmentationDataset(
        f"{DATASET_ROOT}/val/images",
        f"{DATASET_ROOT}/val/masks",
        img_transform=val_transform  # NO Augmentation
    )
    
    # OPTIMIZATION: Use multiple workers and pinned memory
    # Colab T4 has 2 vCPUs, so 2 workers is optimal. 4 causes warnings/overhead.
    NUM_WORKERS = 2 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=(NUM_WORKERS > 0)
    )
    
    # Sanity check: Validate NUM_CLASSES on the first batch
    try:
        print("Checking first batch for class label consistency...")
        check_images, check_masks = next(iter(train_loader))
        unique_vals = torch.unique(check_masks)
        print(f"Unique classes found in first batch: {unique_vals}")
        max_val = unique_vals.max().item()
        if max_val >= NUM_CLASSES:
            print(f"\n{'='*40}")
            print(f"⚠️  CRITICAL WARNING: Found label {max_val} >= NUM_CLASSES ({NUM_CLASSES})")
            print(f"⚠️  Please update NUM_CLASSES in the code to at least {max_val + 1}")
            print(f"{'='*40}\n")
        else:
            print(f"✅ Class labels match NUM_CLASSES ({NUM_CLASSES})")
    except Exception as e:
        print(f"⚠️ Could not validate classes (Dataset might be empty or loading failed): {e}")

    # Validating NUM_CLASSES on first batch...
    # (Output omitted for brevity)

    # ========== CLASS WEIGHTS (Calculated from Phase 1 Analysis) ==========
    # Class 700 is extremely rare (0.08%), Class 10000 dominates (37.65%)
    class_counts = [
        59016700,   # 100
        97779781,   # 200
        311208907,  # 300
        18074830,   # 500
        72049226,   # 550
        45587751,   # 600
        1263280,    # 700! (Severe Imbalance)
        19738472,   # 800
        401180128,  # 7100
        619502525   # 10000
    ]
    
    # Calculate weights: Total / (NumClasses * ClassCount)
    total_pixels = sum(class_counts)
    weights = [total_pixels / (NUM_CLASSES * c) for c in class_counts]
    
    # Clamp extreme weights to prevent instability (e.g., class 700 weight ~130)
    # A standard practice is to dampen the weights or clamp max value.
    # Let's verify standard PyTorch weighting first.
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    print(f"Using Class Weights: {class_weights}")

    # Model, loss, optimizer
    model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
    
    # UPGRADED: Use Dice+CE combined loss for direct IoU optimization
    criterion = CombinedLoss(ce_weight=class_weights, ce_ratio=0.5, dice_ratio=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # NEW: LR Scheduler - Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # ========== RESUME LOGIC ==========
    start_epoch = 0
    best_val_loss = float('inf') # Initialize default
    resume_path = f"{CHECKPOINT_DIR}/last_model.pth"
    
    if os.path.exists(resume_path):
        print(f"🔄 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   Resuming from Epoch {start_epoch+1}")
    else:
        print("🚀 Starting new training run")
    
    # Training loop
    train_losses, val_losses = [], []
    
    for epoch in range(start_epoch, NUM_EPOCHS):
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
        
        # Step the LR scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoints
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # 1. Save Last Model (for resuming)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, f"{CHECKPOINT_DIR}/last_model.pth")

        # 2. Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{CHECKPOINT_DIR}/training_curve.png")
    plt.close()
    
    print(f"Training completed! Best model saved to {CHECKPOINT_DIR}/best_model.pth")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    torch.multiprocessing.freeze_support()
    print(f"Using Device: {DEVICE}")
    main()