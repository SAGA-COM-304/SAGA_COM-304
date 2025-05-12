import os
import torch
from cosmos_tokenizer.image_lib import ImageTokenizer as BaseImageTokenizer
from huggingface_hub import snapshot_download


class ImageTokenizer:
    def __init__(self,
            model_name: str,
            device: torch.device,
            cache_dir: str = ".cache"
    ):
        """
        Initializes the ImageTokenizer with the specified model name and device.

        Arguments:
            model_name (str): Name of the image tokenizer model to use.
                Cosmos-0.1-Tokenizer-DI8x8 (
                Cosmos-0.1-Tokenizer-DI16x16 (
            device (torch.device): Device on which the computation will occur.
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        encoder_path = os.path.join(self.cache_dir, model_name, 'encoder.jit')
        decoder_path = os.path.join(self.cache_dir, model_name, 'decoder.jit')
        snapshot_download(repo_id=f"nvidia/{model_name}", local_dir=os.path.join(cache_dir, model_name))
        self.encoder = BaseImageTokenizer(checkpoint_enc=encoder_path)
        self.decoder = BaseImageTokenizer(checkpoint_dec=decoder_path)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encodes an input image into its latent representation using a pre-trained
        encoder model. The image is converted to RGB format, transformed into a
        tensor, and processed to produce the latent vector.

        Parameters:
            image : torch.Tensor
                The input image tensor to be encoded. (Batch, C, H, W)

        Returns:
            torch.Tensor
                The latent representation of the input image.
        """
        input_tensor = image.to(self.device)
        latent, _ = self.encoder.encode(input_tensor)
        return latent.cpu()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation into its original tensor form using the decoder.

        The method takes a latent tensor representation as input, processes it through the
        decoder, and returns the reconstructed tensor in CPU memory.

        Args:
            latent (torch.Tensor): A tensor representing the latent space representation.
                It should be moved to the appropriate computation device before decoding.

        Returns:
            torch.Tensor: The reconstructed tensor in its original form, moved back to
            the CPU memory.
        """
        output_tensor = self.decoder.decode(latent.to(self.device))
        return output_tensor.cpu()