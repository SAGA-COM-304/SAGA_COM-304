import os
import pytest
import torch
from nanofm.data.tokenizers.dataset import MyImageDataset
from nanofm.data.tokenizers.image_tokenizer import ImageTokenizer

@pytest.fixture
def dataset():
    """
    Fixture to provide a sample image dataset for testing.
    """
    data_path = '/work/com-304/SAGA/raw'
    csv_file = '/home/bousquie/COM-304-FM/SAGA_COM-304/.local_cache/small_vgg.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return MyImageDataset(data_path=data_path, csv_file=csv_file, device=device)

@pytest.fixture
def image_tokenizer():
    """
    Fixture to provide an image tokenizer for testing.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return ImageTokenizer(model_name="Cosmos-0.1-Tokenizer-DI16x16", device=device)


def test_tokenization(dataset, image_tokenizer):
    """
    Test that the image tokenizer encodes and decodes images with expected shapes.
    """
    sample = dataset[0]
    tokens = image_tokenizer.encode(sample['rgb'])
    assert tokens.shape == (1, 16, 16)  # Adjust based on your tokenizer's output shape

    decoded_image = image_tokenizer.decode(tokens)
    assert decoded_image.shape == (1, 3, 256, 256)  # Adjust based on your tokenizer's output shape
    
