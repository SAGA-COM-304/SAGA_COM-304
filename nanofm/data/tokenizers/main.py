import torch
import numpy as np
from torch.utils.data import DataLoader
from nanofm.data.tokenizers.data_loader import MyImageDataset
from nanofm.data.tokenizers.image_tokenizer import ImageTokenizer
from tqdm import tqdm  # Import tqdm for progress bar
import time  # Import time to measure elapsed time


def main():
    dataset_path = "/work/com-304/SAGA/raw"
    csv_path = "/work/com-304/SAGA/vggsound.csv"

    # Initialize dataset and dataloader
    start_time = time.time()
    img_dataset = MyImageDataset(
        data_path=dataset_path,
        csv_file=csv_path
    )
    tqdm.write(f"Dataset initialized in {time.time() - start_time:.2f} seconds")


    dataloader = DataLoader(img_dataset, batch_size=8, shuffle=False)

    # Initialize the image tokenizer
    img_tokenizer = ImageTokenizer(model_name="Cosmos-0.1-Tokenizer-DI8x8", device=torch.device("cuda"))

    # Process a batch of images with a progress bar
    for batch in tqdm(dataloader, desc="Processing batches"):
        start_time = time.time()
        images = batch['rgb'].to(torch.device("cuda"))  # Assuming 'rgb' contains image tensors
        tokenized_images = img_tokenizer.encode(images)  # Tokenize the batch of images
        tqdm.write(f"Tokenization processed in {time.time() - start_time:.2f} seconds")
        print("Tokenized images:", tokenized_images)
        print("Tokenized images shape:", tokenized_images.shape)
        break  # Remove this if you want to process the entire dataset


if __name__ == '__main__':
    main()