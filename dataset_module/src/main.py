import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from prepare import MyImageDataset


def worker_init_fn(worker_id: int):
    torch.set_num_threads(1)
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def parse_args():
    parser = argparse.ArgumentParser(description="Load dataset and iterate batches")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Root folder with 'video' and 'audio' subdirs")
    parser.add_argument('--csv_file', type=str, required=True,
                        help="Path to CSV metadata file")
    parser.add_argument('--file_column', type=str, required=True,
                        help="Column name for clip filenames")
    parser.add_argument('--class_column', type=str, required=True,
                        help="Column name for class labels")
    parser.add_argument('--hard_data', action='store_true',
                        help="Apply hard data augmentation")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of DataLoader workers")
    return parser.parse_args()


def main():
    """
    Main entry point for dataset loading and batch iteration.
    For the moment this function allows us to play with the dataloader. 
    Later it will be used to pretokenize the data.

    This function parses command-line arguments to:
      1. Create an instance of MyImageDataset using provided paths and CSV settings.
      2. Wrap the dataset in a PyTorch DataLoader with specified batch size and workers.
      3. Print the dataset size and the shape of each batch's frames tensor.

    Usage example:
        $ python main.py \
          --data_path /path/to/segments \
          --csv_file /path/to/vggss.csv \
          --file_column video_clip_name \
          --class_column class \
          [--hard_data] \
          [--batch_size 32] \
          [--num_workers 4]

    Command-line arguments:
        data_path (str): Directory path containing segment frames.
        csv_file (str): CSV file path listing video clips and their labels.
        file_column (str): Column name in CSV for filenames.
        class_column (str): Column name in CSV for class labels.
        hard_data (bool): If set, apply hard data preprocessing.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
    """

    args = parse_args()

    dataset = MyImageDataset(
        data_path=args.data_path,
        csv_file=args.csv_file,
        file_column=args.file_column,
        class_column=args.class_column,
        hard_data=args.hard_data
    )
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    for i, batch in enumerate(loader):
        print(f"Batch {i}: frames {batch['frames'].shape}")


if __name__ == '__main__':
    main()