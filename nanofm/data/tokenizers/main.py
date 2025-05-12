import sys
import os
from pathlib import Path
current_file = Path(__file__).resolve()
nanofm_root = current_file.parent.parent.parent 
sys.path.insert(0, str(nanofm_root))
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
from data.tokenizers.data_loader import MyImageDataset
from data.tokenizers.image_tokenizer import ImageTokenizer
from data.tokenizers.audio_tokenizer import AudioTokenizer
from data.tokenizers.video_tokenizer import VideoTokenizer
from transformers import AutoModel, AutoImageProcessor

# NORMAL_REPO  = "alexsax/omnidata_models"
# NORMAL_ENTRY = "surface_normal_dpt_hybrid_384"

# ── helpers ────────────────────────────────────────────────────────────── #
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

def save_npy(dir_: Path, key: str, arr: np.ndarray):
    dir_.mkdir(parents=True, exist_ok=True)
    np.save(dir_ / f"{key}.npy", arr.astype(np.int16))


# ── main ───────────────────────────────────────────────────────────────── #
def main():
    dataset_path = "/work/com-304/SAGA/raw"
    csv_path = "/work/com-304/SAGA/balanced_vggsound.csv"
    image_model_name = "Cosmos-0.1-Tokenizer-DI8x8"
    video_model_name = "Cosmos-0.1-Tokenizer-DV4x8x8"

    ap = argparse.ArgumentParser("tokenise RGB + depth + audio + video")
    ap.add_argument("--data_root", default=dataset_path, type=Path,
                    help="raw/{videos,audios} + csv")
    ap.add_argument("--csv", default=csv_path, type=Path)
    ap.add_argument("--output", default="tokens", type=Path)
    ap.add_argument("--image_model", default=image_model_name, type=str,
                    help="Cosmos tokenizer checkpoint name for images")
    ap.add_argument("--video_model", default=video_model_name, type=str,
                    help="Cosmos tokenizer checkpoint name for videos")
    ap.add_argument("--batch", default=8, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--disable_video", action="store_true", 
                    help="Disable video tokenization to save memory/time")
    ap.add_argument("--group_column", default="group_name", type=str,
                    help="Column name in CSV that contains group (train/eval/test)")
    args = ap.parse_args()

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        sys.exit(f"{args.output} already exists; use --overwrite")

    # Dataset & loader ---------------------------------------------------- #
    #print("Loading dataset...")
    ds = MyImageDataset(data_path=str(args.data_root),
                        csv_file=str(args.csv),
                        group_column=args.group_column,
                        device=torch.device(args.device)) 
    #print("Dataset loaded.")          
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # Tokenizers ---------------------------------------------------------- #
    img_tok = ImageTokenizer(model_name=args.image_model,
                             device=torch.device(args.device))
    aud_tok = AudioTokenizer(device="cpu")
    
    # Tokenizer vidéo (optionnel)
    vid_tok = None
    if not args.disable_video:
        #print("Loading video tokenizer...")
        vid_tok = VideoTokenizer(model_name=args.video_model,
                                device=torch.device(args.device))
        #print("Video tokenizer loaded.")

    # Normal-map model --------------------------------------
    # proc_norm = AutoImageProcessor.from_pretrained(NORMAL_REPO, subfolder=NORMAL_ENTRY)
    # model_norm = (AutoModel.from_pretrained(NORMAL_REPO, subfolder=NORMAL_ENTRY)
    #               .to(args.device).eval())

    # Output dirs par groupe --------------------------------------------- #
    # On créera les dossiers dynamiquement selon les groupes trouvés
    group_dirs = {}

    # Loop ---------------------------------------------------------------- #
    MEAN_d = MEAN.to(args.device)
    STD_d  = STD.to(args.device)

    for batch in dl : # tqdm(dl, desc="Tokenising"):
        keys  : List[str] = batch["ids"]
        rgb_n = batch["rgb"].to(args.device)          
        depth = batch["depth"].to(args.device)        
        audio = batch["audios"]
        frames = batch["frames"].to(args.device) if not args.disable_video else None
        groups = batch["groups"]

        # 1. RGB → de-normalize then encode --------------------------------
        rgb = rgb_n * STD_d + MEAN_d                  
        rgb_codes = img_tok.encode(rgb)               

        # 2. Depth → duplicate 3 channels then encode -----------------------
        depth_3c  = depth.repeat(1, 3, 1, 1)
        depth_codes = img_tok.encode(depth_3c)

        # 3. Normal-map ----------------------------------------
        # with torch.no_grad():
        #     inp = proc_norm(images=(rgb), return_tensors="pt").to(args.device)
        #     n_pred = model_norm(**inp).last_hidden_state
        #     n_pred = F.interpolate(n_pred, size=(256, 256),
        #                            mode="bilinear", align_corners=False)
        # n_img = (n_pred + 1) / 2
        # normal_codes = img_tok.encode(n_img)

        # 4. Audio ----------------------------------------------------------
        audio_codes = [aud_tok.encode(a) for a in audio]

        # 5. Video ------------------------------------------------
        video_codes = None
        if vid_tok is not None and frames is not None:
            # Denormalize frames for video 
            frames_denorm = frames * STD_d.unsqueeze(2) + MEAN_d.unsqueeze(2)
            video_codes = vid_tok.encode(frames_denorm)

        # 6. Save per sample -----------------------------------------------
        for i, (k, group) in enumerate(zip(keys, groups)):
            # Create folder for the group
            if group not in group_dirs:
                group_dirs[group] = {
                    'rgb': args.output / group / f"tok_rgb@256",
                    'depth': args.output / group / f"tok_depth@256",
                    'audio': args.output / group / "tok_audio",
                    'video': args.output / group / "tok_video" if not args.disable_video else None
                }
                
                # Create folder
                for key, path in group_dirs[group].items():
                    if path is not None:
                        path.mkdir(parents=True, exist_ok=True)
            
            # Save in the good group folder
            save_npy(group_dirs[group]['rgb'], k, rgb_codes[i].cpu().numpy())
            save_npy(group_dirs[group]['depth'], k, depth_codes[i].cpu().numpy())
            # save_npy(group_dirs[group]['normal'], k, normal_codes[i].cpu().numpy())
            save_npy(group_dirs[group]['audio'], k, audio_codes[i].cpu().numpy())
            
            # saves video tokens
            if video_codes is not None and group_dirs[group]['video'] is not None:
                save_npy(group_dirs[group]['video'], k, video_codes[i].cpu().numpy())


if __name__ == "__main__":
    main()
