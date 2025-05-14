import os
import shutil
import numpy as np
import torch
from PIL import Image
import pytest
from nanofm.data.utils import save_image

TEST_DIR = "test_output_images"

@pytest.fixture(autouse=True)
def cleanup_test_dir():
    # Setup: create test dir if needed
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    # Teardown: remove test dir and its contents
    shutil.rmtree(TEST_DIR, ignore_errors=True)

def random_image_array(shape=(3, 16, 16)):
    # Values in [0, 1]
    return np.random.rand(*shape).astype(np.float32)

def test_save_image_with_numpy():
    arr = random_image_array()
    save_path = os.path.join(TEST_DIR, "test_numpy.jpg")
    save_image(arr, save_path)
    assert os.path.exists(save_path)
    img = Image.open(save_path)
    assert img.size == (16, 16)
    assert img.mode == "RGB"

def test_save_image_with_tensor():
    tensor = torch.rand(3, 8, 8)
    save_path = os.path.join(TEST_DIR, "test_tensor.jpg")
    save_image(tensor, save_path)
    assert os.path.exists(save_path)
    img = Image.open(save_path)
    assert img.size == (8, 8)
    assert img.mode == "RGB"

def test_save_image_clips_values():
    arr = np.ones((3, 4, 4), dtype=np.float32) * 2.0  # Out of range
    save_path = os.path.join(TEST_DIR, "test_clip.jpg")
    save_image(arr, save_path)
    img = Image.open(save_path)
    np_img = np.array(img)
    assert np.all(np_img == 255)

def test_save_image_creates_directory():
    nested_dir = os.path.join(TEST_DIR, "nested", "dir")
    save_path = os.path.join(nested_dir, "img.jpg")
    arr = random_image_array((3, 5, 5))
    save_image(arr, save_path)
    assert os.path.exists(save_path)
    img = Image.open(save_path)
    assert img.size == (5, 5)