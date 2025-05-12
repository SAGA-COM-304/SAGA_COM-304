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

# Ces imports devraient maintenant fonctionner
from data.tokenizers.data_loader import MyImageDataset
from data.tokenizers.image_tokenizer import ImageTokenizer
from data.tokenizers.audio_tokenizer import AudioTokenizer
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
    model_name = "Cosmos-0.1-Tokenizer-DI8x8"

    ap = argparse.ArgumentParser("tokenise RGB + depth + audio")
    ap.add_argument("--data_root", default=dataset_path, type=Path,
                    help="raw/{videos,audios} + csv")
    ap.add_argument("--csv", default=csv_path, type=Path)
    ap.add_argument("--output", default="tokens", type=Path)
    ap.add_argument("--model", default=model_name, type=str,
                    help="Cosmos tokenizer checkpoint name")
    ap.add_argument("--batch", default=8, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        sys.exit(f"{args.output} already exists; use --overwrite")

    # Dataset & loader ---------------------------------------------------- #
    print("Loading dataset...")
    ds = MyImageDataset(data_path=str(args.data_root),
                        csv_file=str(args.csv)) 
    print("Dataset loaded.")          
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # Tokenizers ---------------------------------------------------------- #
    img_tok = ImageTokenizer(model_name=args.model,
                             device=torch.device(args.device))
    aud_tok = AudioTokenizer(device="cpu")

    # Normal-map model --------------------------------------
    # proc_norm = AutoImageProcessor.from_pretrained(NORMAL_REPO, subfolder=NORMAL_ENTRY)
    # model_norm = (AutoModel.from_pretrained(NORMAL_REPO, subfolder=NORMAL_ENTRY)
    #               .to(args.device).eval())

    # Output dirs --------------------------------------------------------- #
    rgb_dir   = args.output / f"tok_rgb@256"
    depth_dir = args.output / f"tok_depth@256"
    # normal_dir = args.output / f"tok_normal@256"
    audio_dir = args.output / "tok_audio"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    # normal_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Loop ---------------------------------------------------------------- #
    MEAN_d = MEAN.to(args.device)
    STD_d  = STD.to(args.device)

    for batch in tqdm(dl, desc="Tokenising"):
        keys  : List[str] = batch["ids"]
        rgb_n = batch["rgb"].to(args.device)          
        depth = batch["depth"].to(args.device)        
        audio = batch["audios"]                       

        # 1. RGB → de-normalize then encode --------------------------------
        rgb = rgb_n * STD_d + MEAN_d                  
        rgb_codes = img_tok.encode(rgb)               

        # 2. Depth → duplicate 3 channels then encode -----------------------
        depth_3c  = depth.repeat(1, 3, 1, 1)
        depth_codes = img_tok.encode(depth_3c)

        # 3. Normal-map (optionnel) ----------------------------------------
        # with torch.no_grad():
        #     inp = proc_norm(images=(rgb), return_tensors="pt").to(args.device)
        #     n_pred = model_norm(**inp).last_hidden_state
        #     n_pred = F.interpolate(n_pred, size=(256, 256),
        #                            mode="bilinear", align_corners=False)
        # n_img = (n_pred + 1) / 2
        # normal_codes = img_tok.encode(n_img)

        # 4. Audio ----------------------------------------------------------
        audio_codes = [aud_tok.encode(a) for a in audio]

        # 5. Save per sample -----------------------------------------------
        for i, k in enumerate(keys):
            save_npy(rgb_dir,   k, rgb_codes[i].cpu().numpy())
            save_npy(depth_dir, k, depth_codes[i].cpu().numpy())
            # save_npy(normal_dir, k, normal_codes[i].cpu().numpy())
            save_npy(audio_dir, k, audio_codes[i].cpu().numpy())

    print("✅  Export finished →", args.output)


if __name__ == "__main__":
    main()