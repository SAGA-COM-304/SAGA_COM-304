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

from nanofm.data.utils import save_frames


class MyImageDataset(Dataset):

    def _has_readable_frames(self, path: str) -> bool:
        """
        True  → OpenCV can decode at least one frame.
        False → the file is empty / truncated / wrong codec / unreadable.
        """
        cap = cv2.VideoCapture(path)
        ok, _ = cap.read()
        cap.release()
        return ok

    def __init__(
        self,
        data_path: str,
        csv_file: str,
        file_column: str = 'video_clip_name',
        ts_column: str = 'timestamp',
        class_column: str = 'class',
        group_column: str = 'group_name',
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
        self.MEAN = (0.48145466, 0.4578275, 0.40821073)
        self.STD = (0.26862954, 0.26130258, 0.27577711)
        self.IMG_SIZE = 256
        self.NUM_FRAMES = 5
        self.DURATION = 3
        self.TIME_SHIFT = 0.5
        self.SAMPLE_RATE = 24_000
        self.TARGET_LEN = self.SAMPLE_RATE * self.DURATION
        self.DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


        self.file_column = file_column
        self.class_column = class_column
        self.ts_column = ts_column
        self.group_column = group_column


        self.video_dir = os.path.join(data_path, 'videos')
        self.audio_dir = os.path.join(data_path, 'audios')

        tqdm.pandas(desc="Loading data")
        df = pd.read_csv(csv_file)
        valid = df.progress_apply(
            lambda row: (
                os.path.isfile(os.path.join(self.video_dir,
                                            f"{row[file_column]}_{row[ts_column]}.mp4"))
                and self._has_readable_frames(os.path.join(self.video_dir,
                                            f"{row[file_column]}_{row[ts_column]}.mp4"))
            ),
            axis=1
        )
        self.df = df[valid].reset_index(drop=True)

        if hard_data:
            self.transform = T.Compose([
                T.RandomResizedCrop((self.IMG_SIZE, self.IMG_SIZE)),
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
                T.Resize((self.IMG_SIZE, self.IMG_SIZE), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=self.MEAN,
                    std=self.STD
                ),
            ])

        self.device = device

        self.depth_proc = AutoImageProcessor.from_pretrained(self.DEPTH_MODEL_NAME, use_fast=True)
        self.depth_model = None
    
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
        fps = total / self.DURATION
        step = max(1, int(self.TIME_SHIFT * fps))

        before = (self.NUM_FRAMES - 1) // 2
        after = self.NUM_FRAMES - 1 - before

        indices = [mid] + \
                  [max(mid - i * step, 0) for i in range(1, before + 1)] + \
                  [min(mid + i * step, total - 1) for i in range(1, after + 1)]

        indices = sorted(indices)
        return video_tensor[:, indices, :, :], mid

    def _init_models(self):
        """Initialize models lazily once and cache them"""
        if not hasattr(self, '_depth_model_initialized'):
            self.depth_model = (
                AutoModelForDepthEstimation
                .from_pretrained(self.DEPTH_MODEL_NAME, torch_dtype=torch.float16)  # Use fp16 for speed
                .to(self.device)
                .eval()
            )
            self._depth_model_initialized = True

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            A dictionary containing the following keys:
            - 'frames': The selected frames of the video. (Shape: (C, T, H, W))
            - 'rgb': The RGB image of the central frame. (Shape: (C, H, W))
            - 'depth': The depth map of the central frame.
            - 'audios': The audio data.
            - 'labels': The class label.
            - 'ids': The video clip name.
            - 'groups': The group name.
        """
        self._init_models()

        row = self.df.iloc[idx]
        name = row[self.file_column]
        ts = row[self.ts_column]
        label = row[self.class_column]
        group = row[self.group_column]

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
        pil_central = raw[central_idx].resize((self.IMG_SIZE, self.IMG_SIZE), Image.BILINEAR)
    
        # Rgb image
        rgb_im = video_tensor[:, central_idx, :, :]
    
        # Depth
        with torch.no_grad():
            inp = self.depth_proc(images=pil_central, return_tensors='pt').to(self.device)
            depth_pred = self.depth_model(**inp).predicted_depth[0]
        depth = F.interpolate(depth_pred.unsqueeze(0).unsqueeze(0), size=(self.IMG_SIZE, self.IMG_SIZE), mode='bilinear',
                              align_corners=False).squeeze(0)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = depth.cpu()
    
        # Load and process audio
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
    
        if sr != self.SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(wav)
    
        if wav.numel() > self.TARGET_LEN:
            wav = wav[:self.TARGET_LEN]
        else:
            pad = self.TARGET_LEN - wav.numel()
            wav = torch.cat([wav, torch.zeros(pad)], dim=0)
    
        return {
            'frames': selected,
            'rgb': rgb_im,
            'depth': depth,
            'audios': wav,
            'labels': label,
            'ids': name,
            'groups': group,
        }

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverses the normalization process applied to a tensor.
        This method takes a normalized tensor and applies the inverse transformation
        to restore the original scale of the data using the stored mean and standard
        deviation values.
        Args:
            tensor (torch.Tensor): The normalized tensor to be unnormalized. Shape: (B, C, H, W)
                where B is the batch size, C is the number of channels, H is the height, and W is the width.
        Returns:
            torch.Tensor: The unnormalized tensor with the same shape as the input.
        """
        
        return tensor * torch.tensor(self.STD).view(1, 3, 1, 1) + torch.tensor(self.MEAN).view(1, 3, 1, 1)
    

if __name__ == "__main__":
    dataset = MyImageDataset(
        data_path='/work/com-304/SAGA/raw',
        csv_file='/home/bousquie/COM-304-FM/SAGA_COM-304/.local_cache/small_vgg.csv',
        device=torch.device('cuda')
    )

    # Try loading a few samples
    for i in range(5):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Frames shape: {sample['frames'].shape}")
        print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  Depth shape: {sample['depth'].shape}")
        print(f"  Audio shape: {sample['audios'].shape}")
        print(f"  Label: {sample['labels']}")
        print(f"  ID: {sample['ids']}")
        print(f"  Group: {sample['groups']}")

    save_path = '/home/bousquie/COM-304-FM/SAGA_COM-304/.local_cache/dataset'
    
    # Save one sample of the dataset to check the unnormalization
    sample = dataset[0]
    rgb = dataset.unnormalize(sample['rgb'])
    save_frames(rgb, save_path)


    
