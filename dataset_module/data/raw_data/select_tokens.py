import pandas as pd
import os
import shutil
from pathlib import Path

def organize_files(csv_path, source_dir, dest_dir):
    """
    Organize files from source directory into train/test/eval structure.
    
    Args:
        csv_path (str): Path to the CSV file
        source_dir (str): Root directory containing modality folders with .npy files
        dest_dir (str): Destination directory where to create the structure
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # List of modalities to process
    modalities = [
        'tok_audio@24_000',
        'tok_depth@256',
        'tok_label',
        'tok_rgb@256',
        'tok_video@256',
        'tok_video_backup@256'
    ]
    
    # Create directory structure
    for group in df['group_name'].unique():
        for modality in modalities:
            Path(os.path.join(dest_dir, group, modality)).mkdir(parents=True, exist_ok=True)
    
    # Copy files to appropriate locations
    for _, row in df.iterrows():
        video_name = row['video_clip_name']
        group = row['group_name']
        
        for modality in modalities:
            source_file = os.path.join(source_dir, group, modality, f"{video_name}.npy")
            dest_file = os.path.join(dest_dir, group, modality, f"{video_name}.npy")
            
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
                print(f"Copied {source_file} to {dest_file}")
            else:
                print(f"Warning: Source file not found: {source_file}")

if __name__ == "__main__":
    # Configure paths
    csv_path = "small_vgg_250522_14_22.csv"
    source_dir = "/work/com-304/SAGA/tokens_16_05"  # Adjust this to your source directory
    dest_dir = "/work/com-304/SAGA/tokens_16_05_small_vgg_250522"  # Adjust this to your desired destination
    
    # Execute organization
    organize_files(csv_path, source_dir, dest_dir)
    
    # Print summary
    print("\nDataset organization complete!")
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")