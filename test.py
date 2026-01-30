import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm

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
CHECKPOINT_DIR = "checkpoints"
NUM_CLASSES = 10  # Must match training
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256, 256)
OUTPUT_DIR = "predictions"

# ========== UNET MODEL (Same as train.py) ==========
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

# ========== TEST DATASET ==========
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.images[idx]

# ========== INFERENCE ==========
def main():
    # Determine test directory (fallback to val/Color_Images if test/Color_Images doesn't exist)
    test_img_dir = f"{DATASET_ROOT}/test/Color_Images"
    if not os.path.exists(test_img_dir):
        print(f"⚠️  Test directory not found at {test_img_dir}")
        print(f"🔄 Falling back to VALIDATION images for inference demonstration: {DATASET_ROOT}/val/Color_Images")
        test_img_dir = f"{DATASET_ROOT}/val/Color_Images"
        output_dir = "predictions_val_demo"
    else:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    
    # Data transforms - SAME AS TRAINING IMAGES
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    
    # Load test dataset
    test_dataset = TestDataset(
        test_img_dir,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint_path = f"{CHECKPOINT_DIR}/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    
    # Run inference
    print("Running inference...")
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Save batch
            for i, pred in enumerate(predictions):
                filename = filenames[i]
                
                # Convert prediction to numpy (uint8)
                pred_np = pred.cpu().numpy().astype(np.uint8)
                
                # Save as PNG
                pred_img = Image.fromarray(pred_np)
                
                # Ensure output filename is png
                out_name = filename.replace('.jpg', '.png').replace('.jpeg', '.png')
                if not out_name.endswith('.png'):
                    out_name += '.png'
                    
                output_path = os.path.join(output_dir, out_name)
                pred_img.save(output_path)
    
    print(f"Inference completed! Predictions saved to {output_dir}/")

if __name__ == "__main__":
    main()