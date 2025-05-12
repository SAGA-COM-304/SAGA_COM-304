import os
import cv2
import torch
import pandas as pd
import torchaudio
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F
from typing import Tuple
import wave
import numpy as np
from tqdm import tqdm
import time

# Constants
TARGET_SR = 24_000
DURATION_SEC = 3
TARGET_LEN = TARGET_SR * DURATION_SEC

# Depth & normal models
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
NORMAL_HUB_REPO = "alexsax/omnidata_models"
NORMAL_ENTRYPOINT = "surface_normal_dpt_hybrid_384"


class MyImageDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        csv_file: str,
        file_column: str = 'video_clip_name',
        ts_column: str = 'timestamp',
        class_column: str = 'class',
        hard_data: bool = False,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initializes an instance of the data loader class for processing video and audio data,
        with optional data augmentation for hard data scenarios. This initializer sets up the
        file directories, validates the data, applies transformations, and configures device
        settings for further processing.

        Arguments:
            data_path (str): Path to the data directory containing 'video' and 'audio' subdirectories.
            csv_file (str): Path to the CSV file containing metadata about video clips, timestamps, and classes.
            file_column (str, optional): Name of the column in the CSV file that specifies video clip names. Defaults to 'video_clip_name'.
            ts_column (str, optional): Name of the column in the CSV file that specifies timestamps. Defaults to 'timestamp'.
            class_column (str, optional): Name of the column in the CSV file that specifies class information. Defaults to 'class'.
            hard_data (bool, optional): Whether to apply data augmentation for hard data scenarios. Defaults to False.
            device (torch.device, optional): Device configuration for model and data processing. Defaults to 'cpu'.
        """
        self.file_column = file_column
        self.class_column = class_column
        self.ts_column = ts_column

        self.video_dir = os.path.join(data_path, 'videos')
        self.audio_dir = os.path.join(data_path, 'audios')

        tqdm.pandas(desc="Loading data")
        df = pd.read_csv(csv_file)
        valid = df.progress_apply(
            lambda row: os.path.isfile(os.path.join(self.video_dir, f"{row[file_column]}_{row[ts_column]}.mp4")),
            axis=1
        )
        self.df = df[valid].reset_index(drop=True)

        self.size = 256
        self.num_frames = 5
        self.duration = DURATION_SEC
        self.time_shift = 0.5

        self.MEAN = (0.48145466, 0.4578275, 0.40821073)
        self.STD = (0.26862954, 0.26130258, 0.27577711)

        if hard_data:
            self.transform = T.Compose([
                T.RandomResizedCrop((self.size, self.size)),
                T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.8),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(
                    mean=self.MEAN,
                    std=self.STD
                ),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.size, self.size), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=self.MEAN,
                    std=self.STD
                ),
            ])

        self.device = device

        self.depth_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME, use_fast=True)
        self.depth_model = (
                AutoModelForDepthEstimation
                .from_pretrained(DEPTH_MODEL_NAME, torch_dtype=torch.float32)
                .to(self.device)
                .eval()
            )

    def __len__(self):
        return len(self.df)

    def select_frames(self, video_tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Selects a fixed number of frames around the midpoint, with time-shift sampling.
        Return : 
        The selected frames of the video and the indice of the 'Rgb image' that will be used alone.
        """
        total = video_tensor.size(1)
        mid = total // 2
        fps = total / self.duration
        step = max(1, int(self.time_shift * fps))

        before = (self.num_frames - 1) // 2
        after = self.num_frames - 1 - before

        indices = [mid] + \
                  [max(mid - i * step, 0) for i in range(1, before + 1)] + \
                  [min(mid + i * step, total - 1) for i in range(1, after + 1)]

        indices = sorted(indices)
        return video_tensor[:, indices, :, :], mid
    

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        name = row[self.file_column]
        ts = row[self.ts_column]
        label = row[self.class_column]

        # Load and transform video frames
        start_time = time.time()
        cap = cv2.VideoCapture(os.path.join(self.video_dir, f'{name}_{ts}.mp4'))
        frames = []
        raw = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw.append(img)  # needed because we compute depth and normal on the raw images.
            frames.append(self.transform(img))
        cap.release()
        video_tensor = torch.stack(frames, dim=1)
        selected, central_idx = self.select_frames(video_tensor)
        print(f"Video frames loaded and transformed in {time.time() - start_time:.2f} seconds")
    
        # Select the frame for depth and normal computation
        pil_central = raw[central_idx].resize((self.size, self.size), Image.BILINEAR)
    
        # Rgb image
        rgb_im = video_tensor[:, central_idx, :, :]
    
        # Depth
        start_time = time.time()
        with torch.no_grad():
            inp = self.depth_proc(images=pil_central, return_tensors='pt').to(self.device)
            depth_pred = self.depth_model(**inp).predicted_depth[0]
        depth = F.interpolate(depth_pred.unsqueeze(0).unsqueeze(0), size=(self.size, self.size), mode='bilinear',
                              align_corners=False).squeeze(0)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = depth.cpu()
        print(f"Depth computed in {time.time() - start_time:.2f} seconds")
    
        # Load and process audio
        start_time = time.time()
        audio_path = os.path.join(self.audio_dir, f'{name}_{ts}.wav')
        with wave.open(audio_path, 'rb') as wav_file:
            # Get audio properties
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sr = wav_file.getframerate()
            n_frames = wav_file.getnframes()
    
            # Read audio data
            raw_data = wav_file.readframes(n_frames)
    
        # Convert bytes to numpy array
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
    
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
    
        # Convert to float and normalize
        wav = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
    
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    
        if wav.numel() > TARGET_LEN:
            wav = wav[:TARGET_LEN]
        else:
            pad = TARGET_LEN - wav.numel()
            wav = torch.cat([wav, torch.zeros(pad)], dim=0)
        print(f"Audio loaded and processed in {time.time() - start_time:.2f} seconds")
    
        return {
            'frames': selected,
            'rgb': rgb_im,
            'depth': depth,
            'audios': wav,
            'labels': label,
            'ids': name,
        }

# TODO : Prendre en compte Audio pas en Mono + Pas sampler a un rate precis