import os
import torch
#from nanofm.data.tokenizers.data_loader import MyImageDataset
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from huggingface_hub import snapshot_download


class VideoTokenizer:
    def __init__(self,
            model_name: str,
            device: torch.device,
            cache_dir: str = ".local_cache",
    ):
        """
        Initializes the VideoTokenizer with the specified model name and device.

        Arguments:
            model_name (str): Name of the image tokenizer model to use.
                Cosmos-0.1-Tokenizer-DV4x8x8,
                Cosmos-0.1-Tokenizer-DV8x8x8,
                Cosmos-0.1-Tokenizer-DV8x16x16,
            device (torch.device): Device on which the computation will occur.
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        encoder_path = os.path.join(self.cache_dir, model_name, 'encoder.jit')
        decoder_path = os.path.join(self.cache_dir, model_name, 'decoder.jit')
        snapshot_download(repo_id=f"nvidia/{model_name}", local_dir=os.path.join(cache_dir, model_name))
        self.encoder = CausalVideoTokenizer(checkpoint_enc=encoder_path)
        self.decoder = CausalVideoTokenizer(checkpoint_dec=decoder_path)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        input_tensor = video.to(self.device)
        latent, _ = self.encoder.encode(input_tensor)
        return latent.cpu()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        output_tensor = self.decoder.decode(latent.to(self.device))
        return output_tensor.cpu()




# if __name__ == "__main__":
#     dataset_path = "/work/com-304/SAGA/raw"
#     csv_path = "/work/com-304/SAGA/small_vgg.csv"

#     img_dataset = MyImageDataset(
#         data_path=dataset_path,
#         csv_file=csv_path
#     )

#     video = img_dataset[0]['frames'].unsqueeze(0)  # Add batch dimension
#     print("Video shape:", video.shape)  # (1, C, T, H, W)
#     print("Video dtype:", video.dtype)  # Should be torch.float32
#     print("Video device:", video.device)  # Should be cuda:0 or cpu
#     print("Video min:", video.min())  # Should be >= 0
#     print("Video max:", video.max())  # Should be <= 1

#     # Initialize the tokenizer
#     vid_tok = VideoTokenizer(model_name="Cosmos-0.1-Tokenizer-DV4x8x8", device=torch.device("cuda"))

#     # Encode the video
#     encoded_video = vid_tok.encode(video)
#     print("Encoded video shape:", encoded_video.shape)  # Should be (1, 4, 8, 8)
#     print("Encoded video dtype:", encoded_video.dtype)  # Should be torch.int64
#     print("Encoded video device:", encoded_video.device)  # Should be cuda:0 or cpu

#     # Decode the video
#     decoded_video = vid_tok.decode(encoded_video)
#     print("Decoded video shape:", decoded_video.shape)  # Should be (1, 3, T, H, W)
#     print("Decoded video dtype:", decoded_video.dtype)  # Should be torch.float32
#     print("Decoded video device:", decoded_video.device)  # Should be cuda:0 or cpu
#     print("Decoded video min:", decoded_video.min())  # Should be >= 0
#     print("Decoded video max:", decoded_video.max())  # Should be <= 1
    

#     def save_video_frames(video: torch.Tensor, path: str):
#         def unnormalize(tensor: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
#             mean = torch.tensor(mean).view(-1, 1, 1)  # Reshape mean to (C, 1, 1)
#             std = torch.tensor(std).view(-1, 1, 1)    # Reshape std to (C, 1, 1)
#             return tensor * std + mean

#         mean = img_dataset.MEAN
#         std = img_dataset.STD

#         from PIL import Image
#         import numpy as np
#         video = video.squeeze(0)  # Remove batch dimension
#         for i in range(video.shape[1]):
#             unnormalized_frame = unnormalize(video[:, i], mean, std)
#             print("Unnormalized frame shape:", unnormalized_frame.shape)
#             print("Unnormalized frame min:", unnormalized_frame.min())
#             print("Unnormalized frame max:", unnormalized_frame.max())
#             unnormalized_frame = unnormalized_frame.permute(1, 2, 0).cpu().numpy()
#             Image.fromarray((unnormalized_frame * 255).astype(np.uint8)).save(f"frame_{path}_{i}.jpg")
    
#     save_video_frames(video, "original")
#     save_video_frames(decoded_video, "decoded")

        


    
