"""
clean_dataset.py
Scans the dataset folders and removes corrupted or unreadable images.
"""

import os
from PIL import Image
from tqdm import tqdm

def clean_dataset(base_path="dataset"):
    """
    Check all images in the dataset and remove corrupted files.
    
    Args:
        base_path: Path to the dataset directory containing 'real' and 'ai_generated' folders
    """
    folders = ["real", "ai_generated"]
    total_removed = 0
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' does not exist. Skipping...")
            continue
        
        print(f"\nChecking images in {folder_path}...")
        
        # Get all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        removed_count = 0
        
        for filename in tqdm(files, desc=f"Scanning {folder}"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Try to open and verify the image
                with Image.open(file_path) as img:
                    img.verify()  # Verify that it's an actual image
                
                # Re-open to ensure it can be fully loaded
                with Image.open(file_path) as img:
                    img.load()
                    
            except Exception as e:
                # If any error occurs, remove the corrupted file
                print(f"\nRemoving corrupted file: {filename} - Error: {str(e)}")
                os.remove(file_path)
                removed_count += 1
                total_removed += 1
        
        print(f"Removed {removed_count} corrupted images from {folder}/")
    
    print(f"\n{'='*50}")
    print(f"Total corrupted images removed: {total_removed}")
    print(f"Dataset cleaning complete!")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Create dataset folders if they don't exist
    os.makedirs("dataset/real", exist_ok=True)
    os.makedirs("dataset/ai_generated", exist_ok=True)
    
    clean_dataset()