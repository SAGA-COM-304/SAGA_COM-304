import os
import torch
from PIL import Image
from cosmos_tokenizer.image_lib import ImageTokenizer as BaseImageTokenizer
from torchvision.transforms import ToTensor, ToPILImage

class ImageTokenizer:
    def __init__(self, model_name: str, device: torch.device):
        """
        Initializes the ImageTokenizer with the specified model name and device.

        Arguments:
            model_name (str): Name of the image tokenizer model to use.
            device (torch.device): Device on which the computation will occur.
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.cache', 'cosmos_tokenizer_checkpoints')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.encoder = BaseImageTokenizer(checkpoint_enc=os.path.join(self.cache_dir, f'{model_name}_encoder.jit'))
        self.decoder = BaseImageTokenizer(checkpoint_dec=os.path.join(self.cache_dir, f'{model_name}_decoder.jit'))

    def tokenize(self, image_path: str) -> torch.Tensor:
        """
        Tokenizes an input image into a latent tensor representation.

        Arguments:
            image_path (str): Path to the input image file to be tokenized.

        Returns:
            torch.Tensor: A latent tensor representation of the input image.
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = ToTensor()(image).unsqueeze(0).to(self.device)
        latent, _ = self.encoder.encode(input_tensor)
        return latent

    def detokenize(self, latent: torch.Tensor) -> Image.Image:
        """
        Detokenizes a latent tensor representation into a PIL Image.

        Arguments:
            latent (torch.Tensor): The latent tensor representation of an image.

        Returns:
            Image.Image: A PIL Image obtained after decoding the latent representation.
        """
        output_tensor = self.decoder.decode(latent)
        output_tensor = output_tensor.squeeze(0).cpu()
        return ToPILImage()(output_tensor)