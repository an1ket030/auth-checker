
import os
import shutil
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Configuration
RAW_DIR = "ml/data/raw"
PROCESSED_DIR = "ml/data/processed"
IMG_SIZE = 256  # Resize to 256, training will crop to 224
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

def setup_dirs():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            os.makedirs(os.path.join(PROCESSED_DIR, split, label), exist_ok=True)

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify() 
        return True
    except:
        return False

def preprocess_image(src_path, dest_path):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
            img.save(dest_path, "JPEG", quality=90)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")

def get_files_from_raw():
    """
    Scans RAW_DIR and attempts to categorize images into 'real' and 'fake' lists.
    This requires knowledge of the specific dataset structure.
    
    Assumption for 'Fake vs Real Medicine (Kaggle)':
    - Contains folders like 'Real' and 'Fake' (case insensitive).
    """
    real_files = []
    fake_files = []

    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                # Heuristic: Check if 'real' or 'fake' is in the folder name
                folder_name = os.path.basename(root).lower()
                if "real" in folder_name:
                    real_files.append(path)
                elif "fake" in folder_name:
                    fake_files.append(path)
                # Extended heuristic for specific datasets can go here
    
    return real_files, fake_files

def split_and_process(file_list, label):
    random.shuffle(file_list)
    total = len(file_list)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    splits = {
        "train": file_list[:train_end],
        "val": file_list[train_end:val_end],
        "test": file_list[val_end:]
    }

    print(f"Processing {label}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

    for split, files in splits.items():
        for src in tqdm(files, desc=f"{label} -> {split}"):
            filename = os.path.basename(src)
            # Create a unique name to avoid collisions from multiple sources
            unique_name = f"{random.randint(1000,9999)}_{filename}"
            dest = os.path.join(PROCESSED_DIR, split, label, unique_name)
            preprocess_image(src, dest)

if __name__ == "__main__":
    print("Setting up directories...")
    setup_dirs()
    
    print("Scanning raw data...")
    real_imgs, fake_imgs = get_files_from_raw()
    
    print(f"Found {len(real_imgs)} Real images and {len(fake_imgs)} Fake images.")
    
    if not real_imgs or not fake_imgs:
        print("Warning: Could not find labeled data in ml/data/raw. Please check folder structure.")
    else:
        split_and_process(real_imgs, "real")
        split_and_process(fake_imgs, "fake")
        print("Preprocessing complete.")
