
import os
import zipfile
import subprocess
import shutil

# Dataset Configuration
DATASETS = [
    {
        "id": "A",
        "slug": "surajkumarjha1/fake-vs-real-medicine-datasets-images",
        "folder": "fake_vs_real"
    },
    {
        "id": "B",
        "slug": "rishgeeky/indian-pharmaceutical-products",
        "folder": "indian_pharma"
    },
    {
        "id": "C",
        "slug": "mohneesh7/indian-medicine-data",
        "folder": "indian_medicine"
    },
    {
        "id": "D",
        "slug": "harshini-t-g-r/counterfeit_med_detection", # Roboflow might need custom download, leaving placeholder
        "folder": "roboflow", 
        "manual": True # Flag for manual or special handling if kaggle fails
    }
]

RAW_DIR = "ml/data/raw"

def check_kaggle_auth():
    """Checks if kaggle.json exists or KAGGLE_USERNAME/KEY env vars are set."""
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_json):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False

def download_datasets():
    if not check_kaggle_auth():
        print("Error: Kaggle API credentials not found.")
        print("Please place 'kaggle.json' in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    for ds in DATASETS:
        if ds.get("manual"):
            print(f"Skipping manual dataset {ds['id']}: {ds['slug']} (Roboflow requires specific URL/Key)")
            continue

        print(f"Downloading Dataset {ds['id']}: {ds['slug']}...")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", ds['slug'], "-p", RAW_DIR, "--unzip"], check=True)
            print(f"Successfully downloaded {ds['slug']}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {ds['slug']}: {e}")
        except FileNotFoundError:
            print("Error: 'kaggle' command not found. Please install via 'pip install kaggle'.")
            return

def organize_raw_data():
    """
    Moves downloaded files into a structured format if needed.
    This is highly dependent on the unzipped structure of each dataset.
    For now, we just ensure the raw directory exists.
    """
    print(f"Datasets downloaded to {RAW_DIR}. Please inspect and run preprocess.py.")

if __name__ == "__main__":
    download_datasets()
    organize_raw_data()
