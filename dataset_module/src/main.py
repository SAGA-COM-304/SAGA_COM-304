import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

try:
    import git stas as sa
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False


from nanofm.data.tokenizers.dataset import MyImageDataset

#Normalisation constant
MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD  = np.array([0.26862954, 0.26130258, 0.27577711])
TARGET_SR = 24_000


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

def show_sample(frames: torch.Tensor,
                rgb: torch.Tensor,
                depth: torch.Tensor,
                normal: torch.Tensor,
                audio: torch.Tensor):
    """
    frames: Tensor[C, T, H, W]
    rgb:    Tensor[C, H, W]
    depth:  Tensor[1, H, W]
    normal: Tensor[3, H, W]
    audio:  Tensor[L]
    """

    # Move to cpu numpy
    T, C, H, W = frames.shape[1], frames.shape[0], frames.shape[2], frames.shape[3]
    frames_np = frames.numpy().transpose(1, 2, 3, 0)  # [T, H, W, C]
    rgb_np    = rgb.numpy().transpose(1, 2, 0)        # [H, W, C]
    depth_np  = depth.squeeze(0).numpy()                         # [H, W]
    normal_np = normal.numpy().transpose(1, 2, 0)     # [H, W, 3]

    # Denormalize RGB for display
    frames_np = (frames_np * STD + MEAN).clip(0, 1)
    rgb_np    = (rgb_np    * STD + MEAN).clip(0, 1)

    # Figure: one row for frames, then rgb/depth/normal
    fig = plt.figure(figsize=(T*2, 6))
    # Plot each of the T frames
    for i in range(T):
        ax = fig.add_subplot(2, T, 1 + i)
        ax.imshow(frames_np[i])
        ax.axis('off')
        ax.set_title(f"Frame {i+1}")
    # Central RGB
    ax = fig.add_subplot(2, T, T+1)
    ax.imshow(rgb_np)
    ax.axis('off')
    ax.set_title("RGB")
    # Depth map (grayscale)
    ax = fig.add_subplot(2, T, T+2)
    ax.imshow(depth_np, cmap='gray')
    ax.axis('off')
    ax.set_title("Depth")
    # Surface normals
    ax = fig.add_subplot(2, T, T+3)
    ax.imshow((normal_np + 1) / 2)  # normals in [–1,1] → map to [0,1]
    ax.axis('off')
    ax.set_title("Normal")
    plt.tight_layout()
    plt.show()

    # Play audio
    if AUDIO_PLAYBACK_AVAILABLE:
        try:
            wav = (audio.cpu().numpy() * 32767).astype(np.int16)
            play = sa.play_buffer(wav, 1, 2, TARGET_SR)
            play.wait_done()
        except Exception as e:
            print("Audio playback failed:", e)
    else:
        print("Audio playback skipped (simpleaudio not available)")

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

    device = torch.device( "cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu"
    )

    dataset = MyImageDataset(
        data_path=args.data_path,
        csv_file=args.csv_file,
        file_column=args.file_column,
        class_column=args.class_column,
        hard_data=args.hard_data,
        device= device
    )

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    for i, batch in enumerate(loader):
        print(f"Batch {i}: frames {batch['frames'].shape}")
        show_sample(batch['frames'][0], batch['rgb'][0], batch['depth'][0], batch['normal'][0], batch['audios'][0]) #Display only the first sample of the batch


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

#--data_path dataset_module/downloads --csv_file dataset_module/data/raw_data/vggss.csv --file_column video_clip_name --class_column class --batch_size 4
#On cluster : 
#--data_path dataset_module/downloads --csv_file dataset_module/data/raw_data/vggss.csv --file_column video_clip_name --class_column class --batch_size 4