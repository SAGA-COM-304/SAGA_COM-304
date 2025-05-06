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

# Constants
TARGET_SR = 16_000
DURATION_SEC = 3
TARGET_LEN = TARGET_SR * DURATION_SEC

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD  = (0.26862954, 0.26130258, 0.27577711)

# Depth & normal models
DEPTH_MODEL_NAME  = "depth-anything/Depth-Anything-V2-Small-hf"
NORMAL_HUB_REPO   = "alexsax/omnidata_models"
NORMAL_ENTRYPOINT = "surface_normal_dpt_hybrid_384"


class MyImageDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        csv_file: str,
        file_column: str,
        class_column: str,
        hard_data: bool = False,
        device : torch.device = torch.device('cpu')
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

        self.video_dir = os.path.join(data_path, 'video_sub')
        self.audio_dir = os.path.join(data_path, 'audio_sub')

        df = pd.read_csv(csv_file)
        valid = df[file_column].apply(
            lambda name: os.path.isfile(os.path.join(self.video_dir, f'{name}.mp4'))
        )
        self.df = df[valid].reset_index(drop=True)

        self.size = 256
        self.num_frames = 5
        self.duration = DURATION_SEC
        self.time_shift = 0.5

        if hard_data:
            self.transform = T.Compose([
                T.RandomResizedCrop((self.size, self.size)),
                T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.8),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(
                    mean=MEAN,
                    std=STD
                ),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.size, self.size), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=MEAN,
                    std=STD
                ),
            ])
        
        self.device = device

        self.depth_proc  = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME, use_fast=True)
        self.depth_model  = None     # will be lazy-loaded in each worker
        self.normal_model = None
        self.normal_tf = T.Compose([
            T.Resize(384, interpolation=Image.BILINEAR),
            T.CenterCrop(384),
            T.ToTensor()
        ])

        

    def __len__(self):
        return len(self.df)

    def _init_models(self):
        # called inside each worker, on first __getitem__
        if self.depth_model is None:
            self.depth_model = (
                AutoModelForDepthEstimation
                .from_pretrained(DEPTH_MODEL_NAME, torch_dtype=torch.float32)
                .to(self.device)
                .eval()
            )
            self.normal_model = (
                torch.hub.load(NORMAL_HUB_REPO, NORMAL_ENTRYPOINT, pretrained=True)
                    .to(self.device)
                    .eval()
            )


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

        self._init_models()

        row = self.df.iloc[idx]
        name = row[self.file_column]
        label = row[self.class_column]

        # Load and transform video frames
        cap = cv2.VideoCapture(os.path.join(self.video_dir, f'{name}.mp4'))
        frames = []
        raw = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw.append(img) #needed cause we compute depth and normal on the raw images.
            frames.append(self.transform(img))
        cap.release()
        video_tensor = torch.stack(frames, dim=1)

        #select the frames of the video
        selected, central_idx = self.select_frames(video_tensor)

        #select the frame for depth and normal computation
        pil_central = raw[central_idx].resize((self.size, self.size), Image.BILINEAR) 

        #Rgb image 
        rgb_im = video_tensor[:,central_idx,:,:]

        # depth
        with torch.no_grad():
            print(f"[DEBUG] self.device = {self.device!r}  (type: {type(self.device)})")
            inp = self.depth_proc(images=pil_central, return_tensors='pt').to(self.device)
            depth_pred = self.depth_model(**inp).predicted_depth[0]
        depth = F.interpolate(depth_pred.unsqueeze(0).unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False).squeeze(0)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = depth.cpu()

        # normals
        with torch.no_grad():
            nin = self.normal_tf(pil_central).unsqueeze(0).to(self.device)
            nout = self.normal_model(nin)[0]
        n = nout / (nout.norm(dim=0, keepdim=True) + 1e-8)
        normal = F.interpolate(n.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False).squeeze(0)
        normal = normal.cpu()

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
            'rgb' : rgb_im, 
            'depth': depth,
            'normal': normal,
            'audios': wav,
            'labels': label,
            'ids': name,
        }


#TODO : Prendre en compte Audio pas en Mono + Pas sampler a un rate precis