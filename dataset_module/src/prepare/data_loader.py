import os
import cv2
import torch
import pandas as pd
import torchaudio
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# Constants
TARGET_SR = 16_000
DURATION_SEC = 3
TARGET_LEN = TARGET_SR * DURATION_SEC


class MyImageDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        csv_file: str,
        file_column: str,
        class_column: str,
        hard_data: bool = False
    ):
        """
        Args:
            data_path: root folder with 'video/' and 'audio/' subfolders
            csv_file: path to CSV with metadata
            file_column: column name for clip filenames (without extension)
            class_column: column name for labels
            hard_data: enable stronger data augmentations
        """
        self.file_column = file_column
        self.class_column = class_column

        self.video_dir = os.path.join(data_path, 'video')
        self.audio_dir = os.path.join(data_path, 'audio')

        df = pd.read_csv(csv_file)
        valid = df[file_column].apply(
            lambda name: os.path.isfile(os.path.join(self.video_dir, f'{name}.mp4'))
        )
        self.df = df[valid].reset_index(drop=True)

        self.size = 256
        self.num_frames = 5
        self.duration = DURATION_SEC
        self.time_shift = 0.25

        if hard_data:
            self.transform = T.Compose([
                T.RandomResizedCrop((self.size, self.size)),
                T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.8),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                ),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.size, self.size), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                ),
            ])

    def __len__(self):
        return len(self.df)

    def select_frames(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Selects a fixed number of frames around the midpoint, with time-shift sampling.
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
        return video_tensor[:, indices, :, :]

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        name = row[self.file_column]
        label = row[self.class_column]

        # Load and transform video frames
        cap = cv2.VideoCapture(os.path.join(self.video_dir, f'{name}.mp4'))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(self.transform(img))
        cap.release()
        video_tensor = torch.stack(frames, dim=1)

        selected = self.select_frames(video_tensor)

        # Load and process audio
        wav, sr = torchaudio.load(os.path.join(self.audio_dir, f'{name}.wav'))
        wav = wav.squeeze(0)
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

        if wav.numel() > TARGET_LEN:
            wav = wav[:TARGET_LEN]
        else:
            pad = TARGET_LEN - wav.numel()
            wav = torch.cat([wav, torch.zeros(pad)], dim=0)

        return {
            'frames': selected,
            'audios': wav,
            'labels': label,
            'ids': name,
        }


