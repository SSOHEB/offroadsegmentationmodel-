import os
import numpy as np
from PIL import Image
from tqdm import tqdm

DATASET_ROOT = "C:/Users/ssohe/Desktop/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset"
TRAIN_MASKS_DIR = os.path.join(DATASET_ROOT, "train", "Segmentation")

def get_unique_values():
    unique_values = set()
    mask_files = os.listdir(TRAIN_MASKS_DIR)
    
    print(f"Scanning {len(mask_files)} masks in {TRAIN_MASKS_DIR}...", flush=True)
    
    count = 0
    for mask_file in tqdm(mask_files):
        if count >= 20: break # check first 20
        count += 1
        if not mask_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        mask_path = os.path.join(TRAIN_MASKS_DIR, mask_file)
        try:
            # Open as PIL image
            mask = Image.open(mask_path)
            # Convert to numpy array
            mask_np = np.array(mask)
            # unexpected shapes handling (if RGB)
            if len(mask_np.shape) > 2:
                # If RGB, we might need to look at unique RGB tuples, 
                # but the error suggested scalar values (100, 200 etc), 
                # implying it might be read as grayscale or is single channel.
                # Let's flatten and see what scalar values exist if we treat it as such or if it's already 1-channel.
                pass 
            
            unique_in_mask = np.unique(mask_np)
            unique_values.update(unique_in_mask.tolist())
        except Exception as e:
            print(f"Error reading {mask_file}: {e}")

    return sorted(list(unique_values))

if __name__ == "__main__":
    values = get_unique_values()
    print("\n" + "="*40)
    print(f"Found {len(values)} unique values:")
    print(values)
    print("="*40 + "\n")
