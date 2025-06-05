import os
import pytest
import torch
from nanofm.data.tokenizers.dataset import MyImageDataset

@pytest.fixture
def dataset():
    data_path = '/work/com-304/SAGA/raw'
    csv_file = '/home/bousquie/COM-304-FM/SAGA_COM-304/.local_cache/small_vgg.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return MyImageDataset(data_path=data_path, csv_file=csv_file, device=device)

def test_dataset_type(dataset):
    sample = dataset[0]
    
    assert isinstance(sample, dict)
    
    assert sample['frames'].shape == (5, 3, 256, 256)
    # Verify all values are between 0 and 1
    assert (sample['frames'] >= 0).all()
    assert (sample['frames'] <= 1).all()

    assert sample['rgb'].shape == (3, 256, 256)
    # Verify all values are between 0 and 1
    assert (sample['rgb'] >= 0).all()
    assert (sample['rgb'] <= 1).all()

