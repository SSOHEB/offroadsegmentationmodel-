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
DATASET_ROOT = "C:/Users/ssohe/Desktop/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset"
NUM_CLASSES = 10  # Mapped from raw values: 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
IMG_SIZE = (256, 256)

# ========== DATASET ==========
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        # Assumes mask has same filename, potentially different extension
        mask_filename = self.images[idx].replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) # Keep original mode first
        
        if self.img_transform:
            image = self.img_transform(image)
        
        # Custom Mask Processing for Resize + Mapping
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        mask_np = np.array(mask)
        
        # Mapping: {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}
        # Initialize with 255 (ignore_index)
        mask_mapped = np.full_like(mask_np, 255, dtype=np.int64)
        
        mapping = {
            100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
            600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
        }
        
        for k, v in mapping.items():
            mask_mapped[mask_np == k] = v
            
        mask = torch.from_numpy(mask_mapped).long()
        if mask.ndim == 3:
            mask = mask.squeeze(0) # (1, H, W) -> (H, W)
                
        # Sanity check for labels (optional warning)
        if hasattr(mask, 'max') and mask.max() >= NUM_CLASSES:
             # Using print sparingly to avoid log spam, consider logging once per run in production
             pass 
        
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

# ========== TRAINING ==========
def calculate_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0

def main():
    # Data transforms
    
    # Image: Resize + Scale to [0,1] + Tensor conversion
    img_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    
    # Mask: Resize (Nearest Neighbor to preserve classes) + Convert to Tensor (Integer)
    # Mask: Resize ONLY (Mapping in Dataset)
    mask_transform = transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST)
    
    # Load datasets
    train_dataset = SegmentationDataset(
        f"{DATASET_ROOT}/train/Color_Images",
        f"{DATASET_ROOT}/train/Segmentation",
        img_transform=img_transform,
        mask_transform=mask_transform
    )
    val_dataset = SegmentationDataset(
        f"{DATASET_ROOT}/val/Color_Images",
        f"{DATASET_ROOT}/val/Segmentation",
        img_transform=img_transform,
        mask_transform=mask_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
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

    # Model, loss, optimizer
    model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
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
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
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
    main()