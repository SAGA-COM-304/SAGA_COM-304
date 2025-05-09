"""
Converts a dataset organised as

    data_root/
    ├─ video/   *.mp4  (or .mov / .avi …)
    └─ audio/   *.wav  (optional – otherwise audio is read from the video)

into th following format of token files:

    output_root/split/
    ├─ tok_rgb@256/   <sample>.npy   
    └─ tok_audio/     <sample>.npy   (only for audio + image at the moment)

MVP: RGB + Audio only.
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Tuple
import cv2                                   
import numpy as np
import torch
from tqdm import tqdm
from audio_tokenizer import AudioTokenizer
from image_tokenizer import ImageTokenizer


# ───────────────────────── helper functions ─────────────────────────────── #

def list_samples(video_dir: Path,
                 audio_dir: Path | None) -> List[Tuple[str, Path, Path | None]]:
    """Return [(key, video_path, audio_path_or_None), …]."""
    pairs: List[Tuple[str, Path, Path | None]] = []
    for vid in sorted(video_dir.glob("*")):
        key = vid.stem
        wav = (audio_dir / f"{key}.wav") if audio_dir else None
        wav = wav if wav and wav.exists() else None
        pairs.append((key, vid, wav))
    return pairs


def split_train_val_test(samples: List[Tuple[str, Path, Path | None]],
                         ratio: Tuple[float, float, float]
                         ) -> dict[str, List[Tuple[str, Path, Path | None]]]:
    random.shuffle(samples)
    n_train = int(ratio[0] * len(samples))
    n_val   = int(ratio[1] * len(samples))
    return {
        "train": samples[:n_train],
        "val":   samples[n_train:n_train + n_val],
        "test":  samples[n_train + n_val:],
    }


def extract_middle_frame(video_path: Path, img_size: int = 256) -> torch.Tensor:
    """Read the middle frame, resize, return float32 tensor in CHW [0,1]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, nb_frames // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


# ───────────────────────────── pipeline ─────────────────────────────── #

def process_sample(key: str,
                   video_path: Path,
                   audio_path: Path | None,
                   img_tok: ImageTokenizer,
                   aud_tok: AudioTokenizer,
                   out_rgb_dir: Path,
                   out_aud_dir: Path,
                   img_size: int) -> None:
    # RGB → codes ----------------------------------------------------------- #
    frame = extract_middle_frame(video_path, img_size).to(img_tok.device)
    rgb_codes = img_tok.tokenize(frame)                     
    np.save(out_rgb_dir / f"{key}.npy",
            rgb_codes.cpu().numpy().astype(np.int16)) #save .npy

    # Audio → codes --------------------------------------------------------- #
    src_for_audio = audio_path if audio_path is not None else video_path
    aud_codes = aud_tok.encode(src_for_audio)               
    np.save(out_aud_dir / f"{key}.npy",
            aud_codes.cpu().numpy().astype(np.int16)) #save .npy


# ──────────────────────────────── CLI ───────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenise (video/, audio/) dataset into .npy files")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="folder containing video/ and audio/")
    parser.add_argument("--output_root", type=Path, required=True,
                        help="destination folder")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--split_ratio", nargs=3, type=float,
                        default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    video_dir = args.data_root / "videos"
    audio_dir = args.data_root / "audios" if (args.data_root / "audios").is_dir() else None
    if not video_dir.is_dir():
        raise ValueError(f"`video/` not found inside {args.data_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_root} is not empty – use --overwrite")

    # enumerate & split ----------------------------------------------------- #
    samples = list_samples(video_dir, audio_dir)
    splits  = split_train_val_test(samples, tuple(args.split_ratio))  # type: ignore

    # instantiate tokenizers once ------------------------------------------ #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizers on {device} …")
    img_tok = ImageTokenizer(device=device)
    aud_tok = AudioTokenizer(device=device)

    # process --------------------------------------------------------------- #
    for split_name, subset in splits.items():
        if not subset:
            continue
        out_rgb = args.output_root / split_name / f"tok_rgb@{args.img_size}"
        out_aud = args.output_root / split_name / "tok_audio"
        out_rgb.mkdir(parents=True, exist_ok=True)
        out_aud.mkdir(parents=True, exist_ok=True)

        for key, vid_path, wav_path in tqdm(subset,
                                            desc=f"{split_name} ({len(subset)})"):
            process_sample(key, vid_path, wav_path,
                           img_tok, aud_tok,
                           out_rgb, out_aud,
                           args.img_size)

    print(f"\n✅  Export finished → {args.output_root}")


if __name__ == "__main__":
    main()

#UTILISATION : 
#python -m data.tokenizers.tokenize_dataset \
# --data_root com-304/SAGA/raw \
#  --output_root /path/to/output_tokens \
# --overwrite