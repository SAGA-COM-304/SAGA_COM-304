# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
from PIL import Image


def infinite_iterator(loader, distributed=False, sampler=None):
    while True:
        if distributed and sampler is not None:
            # Reset the sampler's epoch to ensure a new shuffle.
            sampler.set_epoch(torch.randint(0, 100000, (1,)).item())
        for batch in loader:
            yield batch


def save_frames(images: torch.Tensor, path: str = ".local_cache/frames"):
    """
    Save a batch of images to disk.
    Args:
        images (torch.Tensor): The images tensor to save. Shape should be (B, C, H, W).
    """
    os.makedirs(path, exist_ok=True)
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (B, H, W, C)
    images = (images * 255).astype(np.uint8)  # Convert to uint8
    for i in range(images.shape[0]):
        Image.fromarray(images[i]).save(f"{path}/frame_{i:04d}.png")