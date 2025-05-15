import sys
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent  
sys.path.insert(0, str(project_root))

import argparse
import tempfile
import time
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
from nanofm.data.tokenizers.dataset import MyImageDataset
from nanofm.data.tokenizers.image_tokenizer import ImageTokenizer
from nanofm.data.tokenizers.audio_tokenizer import AudioTokenizer
from nanofm.data.tokenizers.video_tokenizer import VideoTokenizer

# ── helpers ────────────────────────────────────────────────────────────── #
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

def save_npy(dir_: Path, key: str, arr: np.ndarray):
    dir_.mkdir(parents=True, exist_ok=True)
    np.save(dir_ / f"{key}.npy", arr.astype(np.int32))

# ── main ───────────────────────────────────────────────────────────────── #
def main():
    dataset_path = "/work/com-304/SAGA/raw"
    csv_path = "nanofm/data/tokenizers/small_vgg.csv"
    image_model_name = "Cosmos-0.1-Tokenizer-DI16x16"
    video_model_name = "Cosmos-0.1-Tokenizer-DV8x8x8"
    output = "tokens"
    batch = 8
    device = "cuda"
    group_column = "group_name"
    modality = "audio"

    ap = argparse.ArgumentParser("tokenise RGB + depth + audio + video")
    ap.add_argument("--data_root", default=dataset_path, type=Path,
                    help="raw/{videos,audios} + csv")
    ap.add_argument("--csv", default=csv_path, type=Path)
    ap.add_argument("--output", default=output, type=Path)
    ap.add_argument("--image_model", default=image_model_name, type=str,
                    help="Cosmos tokenizer checkpoint name for images")
    ap.add_argument("--video_model", default=video_model_name, type=str,
                    help="Cosmos tokenizer checkpoint name for videos")
    ap.add_argument("--batch", default=batch, type=int)
    ap.add_argument("--device", default=device, type=str)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--group_column", default=group_column, type=str,
                    help="Column name in CSV that contains group (train/eval/test)")
    ap.add_argument("--modalities", default=modality, type=str)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip files that already exist in output directory")
    
    args = ap.parse_args()

    if args.modalities == "all":
        selected_modalities = {"rgb", "depth", "audio", "video"}
    else:
        selected_modalities = {args.modalities}
    
    print(f"Selected modalities: {selected_modalities}")

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        if not args.skip_existing:
            sys.exit(f"{args.output} already exists; use --overwrite or --skip_existing")

    # Dataset & loader ---------------------------------------------------- #
    print("Loading dataset...")
    ds = MyImageDataset(data_path=str(args.data_root),
                        csv_file=str(args.csv),
                        group_column=args.group_column,
                        device=torch.device(args.device)) 
    print("Dataset loaded.")          
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # Initialize only needed tokenizers ---------------------------------- #
    tokenizers = {}
    
    if "rgb" in selected_modalities or "depth" in selected_modalities:
        print("Loading image tokenizer...")
        tokenizers["image"] = ImageTokenizer(model_name=args.image_model,
                                           device=torch.device(args.device))
        print("Image tokenizer loaded.")
    
    if "audio" in selected_modalities:
        print("Loading audio tokenizer...")
        audio_device = "cpu"
        tokenizers["audio"] = AudioTokenizer(device=audio_device)
        print("Audio tokenizer loaded.")
    
    if "video" in selected_modalities:
        print("Loading video tokenizer...")
        tokenizers["video"] = VideoTokenizer(model_name=args.video_model,
                                           device=torch.device(args.device))
        print("Video tokenizer loaded.")

    # Output dirs par groupe --------------------------------------------- #
    group_dirs = {}

    # Precompute device tensors for normalization
    MEAN_d = MEAN.to(args.device)
    STD_d  = STD.to(args.device)

    # Loop ---------------------------------------------------------------- #
    total_processed = 0
    for batch in tqdm(dl, desc="Tokenizing"):
        keys: List[str] = batch["ids"]
        groups = batch["groups"]
        
        # Create group directories as needed
        for group in groups:
            if group not in group_dirs:
                group_dirs[group] = {}
                for modality in selected_modalities:
                    if modality == "rgb":
                        group_dirs[group]["rgb"] = args.output / group / f"tok_rgb@256"
                    elif modality == "depth":
                        group_dirs[group]["depth"] = args.output / group / f"tok_depth@256"
                    elif modality == "audio":
                        group_dirs[group]["audio"] = args.output / group / "tok_audio"
                    elif modality == "video":
                        group_dirs[group]["video"] = args.output / group / "tok_video"
                
                # Create directories
                for path in group_dirs[group].values():
                    path.mkdir(parents=True, exist_ok=True)

        # Process each modality ----------------------------------------
        batch_tokens = {mod: [] for mod in selected_modalities}
        
        # 1. RGB tokenization
        if "rgb" in selected_modalities:
            print(f"Processing RGB... (batch size: {len(keys)})")
            rgb_n = batch["rgb"].to(args.device)          
            rgb = rgb_n * STD_d + MEAN_d  # Denormalize
            batch_tokens["rgb"] = tokenizers["image"].encode(rgb)
        
        # 2. Depth tokenization  
        if "depth" in selected_modalities:
            print(f"Processing Depth... (batch size: {len(keys)})")
            depth = batch["depth"].to(args.device)        
            depth_3c = depth.repeat(1, 3, 1, 1)  # Convert to 3 channels
            batch_tokens["depth"] = tokenizers["image"].encode(depth_3c)
        
        # 3. Audio tokenization
        if "audio" in selected_modalities:
            print(f"Processing Audio... (batch size: {len(keys)})")
            audio = batch["audios"]
            batch_tokens["audio"] = [tokenizers["audio"].encode(a) for a in audio]
        
        # 4. Video tokenization
        if "video" in selected_modalities:
            print(f"Processing Video... (batch size: {len(keys)})")
            frames = batch["frames"].to(args.device)
            # Denormalize frames
            frames_denorm = frames * STD_d.unsqueeze(2) + MEAN_d.unsqueeze(2)
            batch_tokens["video"] = tokenizers["video"].encode(frames_denorm)

        # Save tokens per sample -------------------------------------------
        for i, (k, group) in enumerate(zip(keys, groups)):
            # Check if file already exists (if skip_existing is set)
            should_skip = False
            if args.skip_existing:
                for modality in selected_modalities:
                    if modality == "audio":
                        # Check for both codes and mask files
                        codes_file = group_dirs[group][modality] / f"{k}_codes.npy"
                        mask_file = group_dirs[group][modality] / f"{k}_mask.npy"
                        if codes_file.exists() and mask_file.exists():
                            should_skip = True
                            break
                    else:
                        file_path = group_dirs[group][modality] / f"{k}.npy"
                        if file_path.exists():
                            should_skip = True
                            break
            
            if should_skip:
                print(f"Skipping {k} (already exists)")
                continue
            
            # Save each modality
            for modality in selected_modalities:
                if modality == "audio":
                    # Audio returns (codes, mask) tuple
                    codes, mask = batch_tokens[modality][i]
                    save_npy(group_dirs[group][modality], f"{k}_codes", codes.cpu().numpy())
                    save_npy(group_dirs[group][modality], f"{k}_mask", mask.cpu().numpy())
                else:
                    # Other modalities are tensors
                    save_npy(group_dirs[group][modality], k, 
                           batch_tokens[modality][i].cpu().numpy())
            
            total_processed += 1

    print(f"\nCompleted! Processed {total_processed} samples.")
    print(f"Output directory: {args.output}")
    
    # Print summary of what was processed
    print("\nModalities processed:")
    for modality in selected_modalities:
        print(f"  ✓ {modality}")


if __name__ == "__main__":
    main()