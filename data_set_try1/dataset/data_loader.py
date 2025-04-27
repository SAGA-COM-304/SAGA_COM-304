#!/usr/bin/env python3
"""
Full pipeline for VGGSound: download, crop, encode, and load as a PyTorch Dataset in one script.

Requirements:
  - yt-dlp installed in PATH
  - ffmpeg installed in PATH
  - torch, torchvision, torchaudio
  - pandas
"""
import os
import csv
import subprocess
import argparse
from multiprocessing import Pool
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchaudio
from torchvision.io import read_video


def download_and_process(args_tuple):
    """
    Download a 5s segment via yt-dlp + ffmpeg, crop video to center square, encode audio.
    args_tuple: (video_id, start_seconds, label, args)
    """
    video_id, start, label, args = args_tuple
    # build URLs and output paths
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_video = args.output_dir / 'video' / f"{video_id}_{start:.2f}.mp4"
    out_audio = args.output_dir / 'audio' / f"{video_id}_{start:.2f}.wav"
    # skip if already exists
    if out_audio.exists() and out_video.exists():
        return

    # get muxed stream URL
    try:
        mux_url = subprocess.check_output([
            'yt-dlp', '-f', f"bestvideo[height>={args.crop_size}]+bestaudio/best", '-g', url
        ], text=True).strip()
    except subprocess.CalledProcessError:
        print(f"yt-dlp failed for {video_id}")
        return

    # ffmpeg command
    cmd = [
        'ffmpeg', '-y', f'-ss', str(start), '-i', mux_url,
        '-t', str(args.duration),
        '-filter:v', f"crop={args.crop_size}:{args.crop_size}:(in_w-{args.crop_size})/2:(in_h-{args.crop_size})/2",
        # video output (no audio)
        '-an', str(out_video),
        # audio output (mono, pcm16le, target_sr)
        '-vn', '-ar', str(args.target_sr), '-ac', '1', '-c:a', 'pcm_s16le', str(out_audio)
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"ffmpeg failed for {video_id}")
        # cleanup partial
        if out_video.exists(): out_video.unlink()
        if out_audio.exists(): out_audio.unlink()


class VGGSoundDataset(Dataset):
    """
    PyTorch Dataset loading pre-processed VGGSound clips.
    Expects directory structure:
      output_dir/
        video/  -> *.mp4
        audio/  -> *.wav
    and a split CSV with columns: video_id, start_seconds, end_seconds, label
    """
    def __init__(self, output_dir, csv_path, split_csv='test.csv', classes_csv='stat.csv', 
                 transforms_audio=None, transforms_video=None):
        self.output_dir = Path(output_dir)
        # load classes
        with open(Path(csv_path) / classes_csv, newline='') as f:
            reader = csv.reader(f)
            self.classes = [row[0] for row in reader]
        # load split file
        self.entries = []  # list of tuples (video_id, start, class_idx)
        with open(Path(csv_path) / split_csv, newline='') as f:
            reader = csv.reader(f)
            for vid, label in reader:
                # approximate start stored in filenames
                # find all matching files
                pattern = f"{vid}_*.wav"
                for wav_path in (self.output_dir / 'audio').glob(pattern):
                    start = float(wav_path.stem.split('_')[1])
                    idx = self.classes.index(label)
                    self.entries.append((vid, start, idx))
        self.transforms_audio = transforms_audio
        self.transforms_video = transforms_video

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        vid, start, class_idx = self.entries[idx]
        base = f"{vid}_{start:.2f}"
        video_path = str(self.output_dir / 'video' / f"{base}.mp4")
        audio_path = str(self.output_dir / 'audio' / f"{base}.wav")

        # load audio
        waveform, sr = torchaudio.load(audio_path)
        # ensure sr matches
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        if self.transforms_audio:
            waveform = self.transforms_audio(waveform)

        # load video (frames, audio, info)
        video_frames, _, _ = read_video(video_path, pts_unit='sec')
        # permute to [C, T, H, W]
        video_tensor = video_frames.permute(3, 0, 1, 2)
        if self.transforms_video:
            video_tensor = self.transforms_video(video_tensor)

        return video_tensor, waveform, class_idx, vid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download & prepare VGGSound clips')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Directory containing stat.csv and split CSVs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed video/ and audio/')
    parser.add_argument('--split', type=str, default='test.csv',
                        help='Name of split CSV (e.g. train.csv)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Duration in seconds to extract')
    parser.add_argument('--target_sr', type=int, default=16000,
                        help='Audio sampling rate')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Video crop size (square)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    (args.output_dir / 'video').mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'audio').mkdir(parents=True, exist_ok=True)

    # Read split CSV rows
    rows = []
    with open(Path(args.csv_dir) / args.split, newline='') as f:
        reader = csv.reader(f)
        for vid, label in reader:
            # start time not in split CSV? If missing, default 0
            start = 0.0
            rows.append((vid, start, label, args))

    # Process downloads in parallel
    with Pool(args.workers) as pool:
        pool.map(download_and_process, rows)

    print('Download and processing complete.')  
    # Example instantiation:
    # dataset = VGGSoundDataset(args.output_dir, args.csv_dir, split_csv=args.split)
    # print(len(dataset))
