import sys
import os
from pathlib import Path
current_file = Path(__file__).resolve()
nanofm_root = current_file.parent.parent.parent 
sys.path.insert(0, str(nanofm_root))
import argparse
import torch.multiprocessing as mp
# import tempfile
# import time
# import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from nanofm.data.tokenizers.dataset import MyImageDataset
from data.tokenizers.image_tokenizer import ImageTokenizer
from data.tokenizers.audio_tokenizer import AudioTokenizer
from data.tokenizers.video_tokenizer import VideoTokenizer
# from transformers import AutoModel, AutoImageProcessor

from concurrent.futures import ThreadPoolExecutor

# NORMAL_REPO  = "alexsax/omnidata_models"
# NORMAL_ENTRY = "surface_normal_dpt_hybrid_384"

# ── helpers ────────────────────────────────────────────────────────────── #
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

def save_npy(dir_: Path, key: str, arr: np.ndarray):
    dir_.mkdir(parents=True, exist_ok=True)
    np.save(dir_ / f"{key}.npy", arr.astype(np.int32))

def worker_init_fn(worker_id: int):
    torch.set_num_threads(1)
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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
    ap.add_argument("--output", default="/work/com-304/SAGA/tokens_13_05", type=Path)
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

    import pandas as pd
    df = pd.read_csv(args.csv)
    groupes = df[args.group_column].unique().tolist()
    group_dirs = {}
    for grp in groupes:
        base = args.output / grp
        group_dirs[grp] = {
            'rgb': base / 'rgb',
            'depth': base / 'depth',
            'audio': base / 'audio',
            'video': base / 'video'
        }
        for sous in group_dirs[grp].values():
            sous.mkdir(parents=True, exist_ok=True)



    # Dataset & loader ---------------------------------------------------- #
    #print("Loading dataset...")
    ds = MyImageDataset(data_path=str(args.data_root),
                        csv_file=str(args.csv),
                        group_column=args.group_column,
                        device=torch.device(args.device)) 
    #print("Dataset loaded.")          
    dl = DataLoader(ds, 
                    batch_size=args.batch,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    worker_init_fn=worker_init_fn,
                    drop_last=False)

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
    # group_dirs = {}

    # Loop ---------------------------------------------------------------- #
    MEAN_d = MEAN.to(args.device)
    STD_d  = STD.to(args.device)

    executor = ThreadPoolExecutor(max_workers=4)
    count = 0
    for batch in dl : # tqdm(dl, desc="Tokenising"):

        futs = []

        keys  : List[str] = batch["ids"]
        rgb_n = batch["rgb"].to(args.device)          
        depth = batch["depth"].to(args.device)        
        audio = batch["audios"].cpu()
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
        
        print(f"Processing batch {count}")
        count += 1
        for i, (k, group) in enumerate(zip(keys, groups)):
            # chemins de sauvegarde
            path_rgb   = group_dirs[group]['rgb']   / f"{k}.npy"
            path_depth = group_dirs[group]['depth'] / f"{k}.npy"
            path_aud   = group_dirs[group]['audio'] / f"{k}.npy"

            # soumettez chaque sauvegarde comme tâche asynchrone
            futs.append(executor.submit(np.save, path_rgb,   rgb_codes[i].cpu().numpy().astype(np.int32)))
            futs.append(executor.submit(np.save, path_depth, depth_codes[i].cpu().numpy().astype(np.int32)))
            futs.append(executor.submit(np.save, path_aud,   audio_codes[i].cpu().numpy().astype(np.int32)))

            if video_codes is not None:
                path_vid = group_dirs[group]['video'] / f"{k}.npy"
                futs.append(executor.submit(np.save, path_vid, video_codes[i].cpu().numpy().astype(np.int32)))

        # 3) (Optionnel) attendez que tout soit écrit avant de passer au batch suivant
        for f in futs:
            f.result()

    # 4) À la fin du main, fermez le pool pour libérer les ressources
    executor.shutdown()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()


