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


def save_image(tensor_or_array, save_path):
    """
    Saves a tensor or numpy array as a JPEG image.
    
    Args:
        tensor_or_array: A tensor or numpy array of shape (3, H, W) with values in [0, 1].
        save_path: Path where the image will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Convert tensor to numpy if needed
    if isinstance(tensor_or_array, torch.Tensor):
        # Move to CPU if on another device
        tensor_or_array = tensor_or_array.detach().cpu()
        # Convert to numpy
        array = tensor_or_array.numpy()
    else:
        array = tensor_or_array
    
    # Ensure array is in the right range
    array = np.clip(array, 0, 1)
    
    # Transpose from (C, H, W) to (H, W, C)
    array = array.transpose(1, 2, 0)
    
    # Scale to [0, 255] and convert to uint8
    array = (array * 255).astype(np.uint8)
    
    # Create a PIL Image and save
    image = Image.fromarray(array)
    image.save(save_path)
    
    print(f"Image saved to {save_path}")