from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

import soundfile
# import librosa

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx):
        audio_path = self.filelist[idx]
        y, sr = torchaudio.load(audio_path)
        y = y.mean(dim=0)  # Convert to mono and remove channel dimension
        y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        
        # Normalize audio manually instead of using sox
        gain = -20 * torch.log10(torch.mean(y ** 2) + 1e-8)
        y = y * (10 ** (gain / 20))
        
        # Handle short audio files
        if y.size(-1) <= self.num_samples:
            y = torch.nn.functional.pad(y, (0, self.num_samples - y.size(-1)))
        else:
            start = torch.randint(0, y.size(-1) - self.num_samples, (1,))
            y = y[start:start + self.num_samples]

        return y
