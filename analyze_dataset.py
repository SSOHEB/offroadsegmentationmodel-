import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
DATASET_ROOT = "C:/Users/ssohe/Desktop/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset"
DIRS_TO_SCAN = [
    ("train", os.path.join(DATASET_ROOT, "train", "Color_Images"), os.path.join(DATASET_ROOT, "train", "Segmentation")),
    ("val",   os.path.join(DATASET_ROOT, "val", "Color_Images"),   os.path.join(DATASET_ROOT, "val", "Segmentation"))
]

# Raw class values expected in the masks
EXPECTED_VALUES = {100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000}
IGNORE_INDEX = 255  # If we were using it, but raw masks shouldn't have it yet effectively

def check_pair(img_name, mask_dir):
    """
    Checks if a corresponding mask exists for the image.
    Logic mirrors train.py: .jpg/.jpeg -> .png
    """
    mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
    mask_path = os.path.join(mask_dir, mask_name)
    if os.path.exists(mask_path):
        return mask_path
    return None

def analyze_dataset():
    print("=== STARTING DATASET ANALYSIS ===")
    
    global_stats = {
        "scanned_images": 0,
        "missing_masks": [],
        "corrupt_files": [],
        "invalid_values": set(),
        "class_counts": defaultdict(int),
        "total_pixels": 0
    }

    for split, img_dir, mask_dir in DIRS_TO_SCAN:
        print(f"\nProcessing {split} split...")
        
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"CRITICAL ERROR: Directory not found: {img_dir} or {mask_dir}")
            continue
            
        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        for img_file in tqdm(images):
            global_stats["scanned_images"] += 1
            img_path = os.path.join(img_dir, img_file)
            
            # 1. Check Pair
            mask_path = check_pair(img_file, mask_dir)
            if not mask_path:
                global_stats["missing_masks"].append(img_path)
                continue
            
            # 2. Check Corruption (Image)
            try:
                with Image.open(img_path) as img:
                    img.verify() 
            except Exception as e:
                global_stats["corrupt_files"].append(f"{img_path} ({str(e)})")
                continue # Skip mask check if image is bad

            # 3. Check Corruption (Mask) & Values
            try:
                # verify() doesn't load data, so we might need load() to check values
                # Re-open for value checking
                with Image.open(mask_path) as mask:
                    mask.load() # Ensure fully readable
                    mask_np = np.array(mask)
                    
                    # flattened unique values
                    uniques, counts = np.unique(mask_np, return_counts=True)
                    
                    for val, count in zip(uniques, counts):
                        if val not in EXPECTED_VALUES:
                            global_stats["invalid_values"].add(int(val))
                        global_stats["class_counts"][int(val)] += int(count)
                        global_stats["total_pixels"] += int(count)
                        
            except Exception as e:
                global_stats["corrupt_files"].append(f"{mask_path} ({str(e)})")

    # --- REPORT GENERATION ---
    print("\n\n" + "="*50)
    print("DATASET HEALTH REPORT")
    print("="*50)
    print(f"Total Images Scanned: {global_stats['scanned_images']}")
    
    print(f"\n[ISSUES]")
    print(f"Missing Masks: {len(global_stats['missing_masks'])}")
    if global_stats["missing_masks"]:
        print(f"  First 5 missing: {global_stats['missing_masks'][:5]}")
        
    print(f"Corrupt Files: {len(global_stats['corrupt_files'])}")
    if global_stats["corrupt_files"]:
        print(f"  First 5 corrupt: {global_stats['corrupt_files'][:5]}")

    print(f"Invalid Label Values Found: {sorted(list(global_stats['invalid_values']))}")
    if global_stats["invalid_values"]:
        print("  ⚠️ WARNING: These values are NOT in your NUM_CLASSES mapping!")

    print(f"\n[CLASS DISTRIBUTION]")
    print(f"{'Class ID':<10} | {'Pixels':<15} | {'Percent':<10}")
    print("-" * 40)
    
    # Sort by class ID
    sorted_cls = sorted(global_stats["class_counts"].items())
    for cls_id, count in sorted_cls:
        pct = (count / global_stats["total_pixels"]) * 100
        print(f"{cls_id:<10} | {count:<15,} | {pct:.2f}%")
        
    print("="*50)

if __name__ == "__main__":
    analyze_dataset()
