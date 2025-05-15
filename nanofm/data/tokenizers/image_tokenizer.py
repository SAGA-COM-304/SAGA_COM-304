import os
import torch
from cosmos_tokenizer.image_lib import ImageTokenizer as BaseImageTokenizer
from huggingface_hub import snapshot_download


class ImageTokenizer:
    def __init__(self,
            model_name: str,
            device: torch.device,
            cache_dir: str = ".local_cache",
    ):
        """
        Initializes the ImageTokenizer with the specified model name and device.

        Arguments:
            model_name (str): Name of the image tokenizer model to use.
                Cosmos-0.1-Tokenizer-CI8x8
                Cosmos-0.1-Tokenizer-CI16x16
                Cosmos-0.1-Tokenizer-DI8x8
                Cosmos-0.1-Tokenizer-DI16x16
            device (torch.device): Device on which the computation will occur.
        """
        self.model_name = model_name
        self.type = "continuous" if "CI" in model_name else "discrete" if "DI" in model_name else None
        if self.type is None:
            raise ValueError(f"Invalid model name: {model_name}. Must contain 'CI' or 'DI'.")
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        encoder_path = os.path.join(self.cache_dir, model_name, 'encoder.jit')
        decoder_path = os.path.join(self.cache_dir, model_name, 'decoder.jit')
        snapshot_download(repo_id=f"nvidia/{model_name}", local_dir=os.path.join(cache_dir, model_name))
        self.tokenizer = BaseImageTokenizer(checkpoint_enc=encoder_path, 
                                            checkpoint_dec=decoder_path, 
                                            device=self.device, 
                                            dtype = "bfloat16"
                                          )


    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encodes an input image tensor into a latent representation or discrete indices.

        Args:
            image (torch.Tensor): The input image tensor to be encoded. Shape: (B, 3, H, W), Range: [0...1].

        Returns:
            torch.Tensor: The encoded representation of the image. If the tokenizer type
                is "continuous", a latent representation is returned. If the tokenizer
                type is "discrete", discrete indices are returned.
                Shape: Cosmos-0.1-Tokenizer-CIhxw -> (B, H/h, W/w)
        """
        # TODO: Adapt for 0..1
        image = image.to(self.device).to(torch.bfloat16)
        if self.type == "continuous":
            (latent,) = self.tokenizer.encode(image)
            return latent
        elif self.type == "discrete":
            (indices, _) = self.tokenizer.encode(image)
            return indices
        

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decodes a tensor of tokens into their original representation.
        Args:
            tokens (torch.Tensor): A tensor containing the tokens to decode.
        Returns:
            torch.Tensor: The decoded tensor in (Shape: (B, 3, H, W)) format. Range: [0...1].
        """
        # TODO: Adapt for 0..1
        return self.tokenizer.decode(tokens.to(self.device).to(torch.bfloat16)).clamp(-1, 1)
    
if __name__ == "__main__":
    from dataset import MyImageDataset

    out_path = ".local_cache/image_tokenizer"
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dataset = MyImageDataset(data_path="/work/com-304/SAGA/raw",
                                    csv_file="/home/bousquie/COM-304-FM/SAGA_COM-304/.local_cache/small_vgg.csv",
                                    device=device)
    img_tok = ImageTokenizer(model_name="Cosmos-0.1-Tokenizer-DI16x16", device=device)

    sample = img_dataset[0]
    encoded_token = img_tok.encode(sample['rgb'])
    assert encoded_token.shape == (1, 256, 256)  # Adjust based on your tokenizer's output shape
    print(encoded_token.max)
    print(encoded_token.min)

    decoded_image = img_tok.decode(encoded_token)
    assert decoded_image.shape == (1, 3, 256, 256)  # Adjust based on your tokenizer's output shape
    
    
