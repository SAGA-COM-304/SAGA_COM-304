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
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import wave


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


def save_audio(waveform: torch.Tensor, filepath: str, sample_rate: int = 24000):
    """
    Save an audio waveform to disk using only built-in libraries.
    
    Args:
        waveform (torch.Tensor): Audio waveform tensor of shape [channels, samples] or [samples]
        filepath (str): Path where to save the audio file
        sample_rate (int, optional): Sampling rate of the audio. Defaults to 24000.
    """
    # Ensure waveform is on CPU and convert to numpy
    waveform = waveform.cpu().numpy()
    
    # Ensure waveform is 2D [channels, samples]
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    
    # Add .wav extension if not present
    if not filepath.endswith('.wav'):
        filepath = filepath + '.wav'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert to int16 format
    waveform = (waveform * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(filepath, 'wb') as wav_file:
        n_channels = waveform.shape[0]
        sampwidth = 2  # 2 bytes for int16
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.T.tobytes())
    
    print(f"Audio saved to {filepath}")


def save_video(frames: torch.Tensor, filepath: str, fps: int = 3):
    """
    Save a gif 
    Args:
        frames (torch.Tensor): Tensor of shape [C, T, H, W]
        filepath (str): Path where to save the video file
        fps (int): Frames per second for the gif
    """
    # Ensure frames are on CPU and convert to numpy
    frames = frames.cpu().numpy()
    
    # Ensure frames are in the right range
    frames = np.clip(frames, 0, 1)
    
    # Transpose from (C, T, H, W) to (T, H, W, C)
    frames = frames.transpose(1, 2, 3, 0)
    
    # Scale to [0, 255] and convert to uint8
    frames = (frames * 255).astype(np.uint8)
    
    # Create a PIL Image and save as gif
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    image = Image.fromarray(frames[0])
    image.save(filepath, save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]], 
               duration=1000/fps, loop=0)
    
    print(f"Video saved to {filepath}")