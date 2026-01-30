import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# Use the same logic as analyze_dataset to find root
DATASET_ROOT = "C:/Users/ssohe/Desktop/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset"
OUTPUT_ROOT = "dataset_256"
TARGET_SIZE = (256, 256)

# Debug Check
if not os.path.exists(DATASET_ROOT):
    if os.path.exists("dataset_root"):
        DATASET_ROOT = "dataset_root"
    else:
        print(f"Error: Could not find dataset at {DATASET_ROOT}")
        exit(1)

DIRS_TO_PROCESS = [
    ("train", os.path.join(DATASET_ROOT, "train", "Color_Images"), os.path.join(DATASET_ROOT, "train", "Segmentation")),
    ("val",   os.path.join(DATASET_ROOT, "val", "Color_Images"),   os.path.join(DATASET_ROOT, "val", "Segmentation"))
]

MAPPING = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def preprocess():
    print(f"=== STARTING PREPROCESSING ===")
    print(f"Input: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Target Size: {TARGET_SIZE}", flush=True)

    for split, img_dir, mask_dir in DIRS_TO_PROCESS:
        # Create output directories
        out_img_dir = os.path.join(OUTPUT_ROOT, split, "images")
        out_mask_dir = os.path.join(OUTPUT_ROOT, split, "masks")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        
        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\nProcessing {split}: {len(images)} images...", flush=True)
        
        for i, img_filename in enumerate(images):
            if i % 100 == 0: print(f"  {i}/{len(images)}", flush=True)

            # Paths
            img_path = os.path.join(img_dir, img_filename)
            mask_filename = img_filename.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                print(f"Skipping missing mask: {img_filename}")
                continue
                
            try:
                # 1. Image: Resize (Bilinear)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img_resized = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
                    # Save as JPG to save space/time (quality 95 is good enough for training)
                    img_resized.save(os.path.join(out_img_dir, img_filename), quality=95)

                # 2. Mask: Resize (Nearest) + Map
                with Image.open(mask_path) as mask:
                    mask = mask.resize(TARGET_SIZE, Image.Resampling.NEAREST)
                    mask_np = np.array(mask)
                    
                    # Pre-map to 0-9
                    # Create a blank array filled with ignore_index (255)
                    mask_mapped = np.full_like(mask_np, 255, dtype=np.uint8)
                    
                    for k, v in MAPPING.items():
                        mask_mapped[mask_np == k] = v
                        
                    # Save as PNG (lossless is required for masks)
                    final_mask = Image.fromarray(mask_mapped)
                    final_mask.save(os.path.join(out_mask_dir, mask_filename))
                    
            except Exception as e:
                print(f"Failed to process {img_filename}: {e}")

    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"New dataset saved at: {os.path.abspath(OUTPUT_ROOT)}")

if __name__ == "__main__":
    preprocess()
