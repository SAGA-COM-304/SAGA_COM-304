"""
Pre-tokenise a raw dataset laid out as

    data_root/
    ├─ video/   clip_0001.mp4
    └─ audio/   clip_0001.wav   (optional)

and produce CLEVR-style token files:

    output_root/{train,val,test}/
       ├─ tok_rgb@<img_size>/ clip_0001.npy   
       └─ tok_audio/          clip_0001.npy   

Only RGB + Audio for now.
"""

from __future__ import annotations
import argparse, random, tempfile, shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .audio_tokenizer import AudioTokenizer          
from .image_tokenizer import ImageTokenizer          


# ────────────────────────── helper functions ───────────────────────────── #

def list_samples(video_dir: Path,
                 audio_dir: Path | None) -> List[Tuple[str, Path, Path | None]]:
    """Return [(key, video_path, audio_path_or_None), …]."""
    out: List[Tuple[str, Path, Path | None]] = []
    for vid in sorted(video_dir.glob("*")):
        key = vid.stem
        wav = (audio_dir / f"{key}.wav") if audio_dir else None
        wav = wav if wav and wav.exists() else None
        out.append((key, vid, wav))
    return out


def split_train_val_test(samples: List[Tuple[str, Path, Path | None]],
                         ratio: Tuple[float, float, float]
                         ) -> dict[str, List[Tuple[str, Path, Path | None]]]:
    random.shuffle(samples)
    n_train = int(ratio[0] * len(samples))
    n_val   = int(ratio[1] * len(samples))
    return {
        "train": samples[:n_train],
        "val"  : samples[n_train:n_train + n_val],
        "test" : samples[n_train + n_val:],
    }


def extract_middle_frame_to_file(video_path: Path,
                                 img_size: int,
                                 tmp_dir: Path) -> Path:
    """Save the middle RGB frame to a tmp .png and return its path."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, nb // 2)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read mid-frame from {video_path}")
    frame_bgr = cv2.resize(frame_bgr, (img_size, img_size),
                           interpolation=cv2.INTER_AREA)
    tmp_path = tmp_dir / f"{video_path.stem}_rgb.png"
    cv2.imwrite(str(tmp_path), frame_bgr)
    return tmp_path


# ───────────────────────── pipeline per sample ─────────────────────────── #

def process_sample(key: str,
                   video_path: Path,
                   audio_path: Path | None,
                   img_tok: ImageTokenizer,
                   aud_tok: AudioTokenizer,
                   out_rgb_dir: Path,
                   out_aud_dir: Path,
                   img_size: int,
                   tmp_dir: Path) -> None:
    # ---- RGB ------------------------------------------------------------- #
    img_file = extract_middle_frame_to_file(video_path, img_size, tmp_dir)
    rgb_codes = img_tok.tokenize(str(img_file))           
    np.save(out_rgb_dir / f"{key}.npy",
            rgb_codes.cpu().numpy().astype(np.int16))

    # ---- Audio ----------------------------------------------------------- #
    src = audio_path if audio_path is not None else video_path
    aud_codes = aud_tok.encode(src)                       
    np.save(out_aud_dir / f"{key}.npy",
            aud_codes.cpu().numpy().astype(np.int16))


# ──────────────────────────────── CLI ──────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        "tokenize_dataset",
        description="Pre-tokenise (video/, audio/) dataset into CLEVR-style .npy")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="folder with video/ and (optional) audio/")
    parser.add_argument("--output_root", type=Path, required=True,
                        help="destination folder")
    parser.add_argument("--img_size", type=int, default=256,
                        help="resize extracted RGB frame to this square size")
    parser.add_argument("--img_model", type=str, default="cosmos_256",
                        help="model name for ImageTokenizer (see image_tokenizer.py)")
    parser.add_argument("--split_ratio", nargs=3, type=float, default=(0.8, 0.1, 0.1),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # reproducibility ------------------------------------------------------ #
    random.seed(args.seed)
    np.random.seed(args.seed)

    # paths ---------------------------------------------------------------- #
    video_dir = args.data_root / "video"
    if not video_dir.is_dir():
        raise ValueError(f"`video/` not found inside {args.data_root}")
    audio_dir = args.data_root / "audio" if (args.data_root / "audio").is_dir() else None

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_root} is not empty – use --overwrite")
    args.output_root.mkdir(parents=True, exist_ok=True)

    # split ---------------------------------------------------------------- #
    samples = list_samples(video_dir, audio_dir)
    splits  = split_train_val_test(samples, tuple(args.split_ratio))  # type: ignore

    # tokenizers ----------------------------------------------------------- #
    device_str = ("cuda" if torch.cuda.is_available()
                  else "cpu")
    print(f"Loading tokenizers on {device_str} …")
    img_tok = ImageTokenizer(model_name=args.img_model,
                             device=torch.device(device_str))
    aud_tok = AudioTokenizer(device=device_str)

    # tmp workspace -------------------------------------------------------- #
    tmp_root = Path(tempfile.mkdtemp(prefix="tok_tmp_"))

    try:
        for split_name, subset in splits.items():
            if not subset:
                continue
            out_rgb = args.output_root / split_name / f"tok_rgb@{args.img_size}"
            out_aud = args.output_root / split_name / "tok_audio"
            out_rgb.mkdir(parents=True, exist_ok=True)
            out_aud.mkdir(parents=True, exist_ok=True)

            for key, v_path, wav_path in tqdm(subset,
                                              desc=f"{split_name} ({len(subset)})"):
                process_sample(key, v_path, wav_path,
                               img_tok, aud_tok,
                               out_rgb, out_aud,
                               args.img_size, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\n✅  Export finished → {args.output_root}")


if __name__ == "__main__":
    main()