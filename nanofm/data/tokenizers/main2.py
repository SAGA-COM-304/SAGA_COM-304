import sys
import os
from pathlib import Path


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent  
sys.path.insert(0, str(project_root))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
from nanofm.data.tokenizers.dataset import MyImageDataset
from nanofm.data.tokenizers.image_tokenizer import ImageTokenizer
from nanofm.data.tokenizers.audio_tokenizer import AudioTokenizer
from nanofm.data.tokenizers.video_tokenizer import VideoTokenizer
from nanofm.data.tokenizers.video_image_tokenizer import VideoImageTokenizer

# ── Configuration ─────────────────────────────────────────────────────── #
DATASET_PATH = "/work/com-304/SAGA/raw"
CSV_PATH = "nanofm/data/tokenizers/vggsound_valid/02.csv"
OUTPUT_PATH = "tokens"
IMAGE_MODEL_NAME = "Cosmos-0.1-Tokenizer-DI16x16"
VIDEO_MODEL_NAME = "Cosmos-0.1-Tokenizer-DV8x8x8"
BATCH_SIZE = 32
DEVICE = "cuda"
GROUP_COLUMN = "group_name"
MODALITIES = "all"  # "all", "rgb", "depth", "audio", "video" or comma-separated like "rgb,audio"
OVERWRITE = False
SKIP_EXISTING = True

# ── helpers ────────────────────────────────────────────────────────────── #
def save_npy(dir_: Path, key: str, arr: np.ndarray):
    """Save numpy array as .npy file"""
    dir_.mkdir(parents=True, exist_ok=True)
    np.save(dir_ / f"{key}.npy", arr.astype(np.uint32))

# ── main ───────────────────────────────────────────────────────────────── #
def main():
    # Parse modalities
    if MODALITIES == "all":
        selected_modalities = {"rgb", "depth", "audios", "video", "video_backup"}
    else:
        selected_modalities = set(MODALITIES.split(","))
    
    print(f"Selected modalities: {selected_modalities}")

    # Check output directory
    output_path = Path(OUTPUT_PATH)
    if output_path.exists() and any(output_path.iterdir()) and not OVERWRITE:
        if not SKIP_EXISTING:
            sys.exit(f"{output_path} already exists; set OVERWRITE=True or SKIP_EXISTING=True")

    # Initialize dataset and dataloader
    print("Loading dataset...")
    ds = MyImageDataset(
        data_path=DATASET_PATH,
        csv_file=CSV_PATH,
        group_column=GROUP_COLUMN,
        device=torch.device(DEVICE)
    )
    print(f"Dataset loaded with {len(ds)} samples.")
    
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize tokenizers based on selected modalities
    tokenizers = {}
    
    if "rgb" in selected_modalities or "depth" in selected_modalities:
        print("Loading image tokenizer...")
        tokenizers["image"] = ImageTokenizer(
            model_name=IMAGE_MODEL_NAME,
            device=torch.device(DEVICE)
        )
        print("Image tokenizer loaded.")
    
    if "audios" in selected_modalities:
        print("Loading audio tokenizer...")
        tokenizers["audios"] = AudioTokenizer(device=torch.device(DEVICE))
        print("Audio tokenizer loaded.")
    
    if "video" in selected_modalities:
        print("Loading video tokenizer...")
        tokenizers["video"] = VideoTokenizer(
            model_name=VIDEO_MODEL_NAME,
            device=torch.device(DEVICE)
        )
        print("Video tokenizer loaded.")

    if "video_backup" in selected_modalities in selected_modalities:
        print("Loading video_backup tokenizer...")
        tokenizers["video_backup"] = VideoImageTokenizer(
            model_name=IMAGE_MODEL_NAME,
            device=torch.device(DEVICE)
        )
        print("Video_backup tokenizer loaded.")
    

    # Create output directories per group
    group_dirs = {}

    # Processing loop
    total_processed = 0
    total_skipped = 0
    
    print("Starting tokenization...")
    for batch_idx, batch in enumerate(tqdm(dl, desc="Processing batches")):
        keys: List[str] = batch["ids"]
        groups = batch["groups"]
        
        # Create group directories if they don't exist
        for group in groups:
            if group not in group_dirs:
                group_dirs[group] = {}
                for modality in selected_modalities:
                    if modality == "rgb":
                        group_dirs[group]["rgb"] = output_path / group / "tok_rgb@256"
                    elif modality == "depth":
                        group_dirs[group]["depth"] = output_path / group / "tok_depth@256"
                    elif modality == "audios":
                        group_dirs[group]["audios"] = output_path / group / "tok_audio@24_000"
                    elif modality == "video":
                        group_dirs[group]["video"] = output_path / group / "tok_video@256"
                    elif modality == "video_backup":
                        group_dirs[group]["video"] = output_path / group / "tok_video_backup@256"
                
                # Create directories
                for path in group_dirs[group].values():
                    path.mkdir(parents=True, exist_ok=True)

        # Process each modality for this batch
        batch_tokens = {}
        
        # 1. RGB tokenization
        if "rgb" in selected_modalities:
            #print(f"Processing RGB batch {batch_idx + 1}...")
            rgb_data = batch["rgb"].to(DEVICE)
            # RGB data from dataset is already in [0, 1] range
            batch_tokens["rgb"] = tokenizers["image"].encode(rgb_data)
        
        # 2. Depth tokenization  
        if "depth" in selected_modalities:
            #print(f"Processing Depth batch {batch_idx + 1}...")
            depth_data = batch["depth"].to(DEVICE)
            # Convert single channel depth to 3 channels for image tokenizer
            if depth_data.dim() == 3:  # (B, H, W)
                depth_data = depth_data.unsqueeze(1)  # (B, 1, H, W)
            depth_3c = depth_data.repeat(1, 3, 1, 1)  # (B, 3, H, W)
            batch_tokens["depth"] = tokenizers["image"].encode(depth_3c)
        
        # 3. Audio tokenization
        if "audios" in selected_modalities:
            #print(f"Processing Audio batch {batch_idx + 1}...")
            audio_data = batch["audios"]
            batch_tokens["audios"] = tokenizers["audios"].encode(audio_data)
                

        # 4. Video tokenization
        if "video" in selected_modalities:
            #print(f"Processing Video batch {batch_idx + 1}...")
            video_data = batch["frames"].to(DEVICE)
            # Video data from dataset is already in [0, 1] range
            batch_tokens["video"] = tokenizers["video"].encode(video_data)
        
        if "video_backup" in selected_modalities:
            video_backup_data = batch["frames"].to(DEVICE)
            batch_tokens["video_backup"] = tokenizers["video_backup"].encode(video_backup_data)

        # Save tokens for each sample in the batch
        for i, (k, group) in enumerate(zip(keys, groups)):
            # Check if files already exist (if skip_existing is set)
            should_skip = False
            if SKIP_EXISTING:
                files_to_check = []
                for modality in selected_modalities:
                    files_to_check.append(group_dirs[group][modality] / f"{k}.npy")
                
                if all(f.exists() for f in files_to_check):
                    should_skip = True
            
            if should_skip:
                #print(f"Skipping {k} (already exists)")
                total_skipped += 1
                continue
            
            # Save each modality
            for modality in selected_modalities:
                tokens = batch_tokens[modality][i]
                save_npy(group_dirs[group][modality], k, 
                       tokens.cpu().numpy())
            
            total_processed += 1

    # Print summary
    print(f"\nTokenization completed!")
    print(f"Total samples processed: {total_processed}")
    print(f"Total samples skipped: {total_skipped}")
    print(f"Output directory: {output_path}")
    
    print("\nModalities processed:")
    for modality in selected_modalities:
        print(f"  ✓ {modality}")
    
    # Print group distribution
    print("\nSamples per group:")
    df = ds.df
    for group in df[GROUP_COLUMN].unique():
        count = len(df[df[GROUP_COLUMN] == group])
        print(f"  {group}: {count} samples")


if __name__ == "__main__":
    main()