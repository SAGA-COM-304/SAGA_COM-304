import os
import torch
from cosmos_tokenizer.image_lib import ImageTokenizer as BaseImageTokenizer
from huggingface_hub import snapshot_download
from torchvision import transforms as T


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

        self.normalize_cosmos = T.Normalize(mean = 0.5,
                                            std = 0.5)
        
        self.denormalize_cosmos = T.Normalize(mean = -1.0,
                                              std = 2.0) 

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
        image = self.normalize_cosmos(image)                #map from [0,1] => [-1,1]
        image = image.to(self.device).to(torch.bfloat16)
        (tokens,) = self.tokenizer.encode(image)
        return tokens
        

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decodes a tensor of tokens into their original representation.
        Args:
            tokens (torch.Tensor): A tensor containing the tokens to decode.
        Returns:
            torch.Tensor: The decoded tensor in (Shape: (B, 3, H, W)) format. Range: [0...1].
        """
        output_decode = self.tokenizer.decode(tokens).clamp(-1, 1).to(torch.float32)
        return self.denormalize_cosmos(output_decode)

if __name__ == "__main__":
    from dataset import MyImageDataset
    from nanofm.data.utils import save_image

    out_path = ".local_cache/image_tokenizer"
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dataset = MyImageDataset(data_path="/work/com-304/SAGA/raw",
                                    csv_file="/home/bousquie/COM-304-FM/SAGA_COM-304/.local_cache/small_vgg.csv",
                                    device=device)
    img_tok = ImageTokenizer(model_name="Cosmos-0.1-Tokenizer-DI16x16", device=device)


    # Test RGB
    # ==== Choose an image
    rgb_sample = img_dataset[0]["rgb"]
    save_image(rgb_sample, os.path.join(out_path, "rgb_sample.jpg"))

    # ==== Print information
    print("RGB sample shape:", rgb_sample.shape)
    print("RGB sample dtype:", rgb_sample.dtype)
    print("RGB sample device:", rgb_sample.device)
    print("RGB sample min:", rgb_sample.min())
    print("RGB sample max:", rgb_sample.max())

    # ==== Encode
    tokens = img_tok.encode(rgb_sample.unsqueeze(0))
    print("Encoded RGB tokens shape:", tokens.shape)
    print("Encoded RGB tokens dtype:", tokens.dtype)
    print("Encoded RGB tokens device:", tokens.device)
    print("Encoded RGB tokens min:", tokens.min())
    print("Encoded RGB tokens max:", tokens.max())    

    # ==== Decode
    rgb_dec_sample = img_tok.decode(tokens)
    print("Decoded RGB sample shape:", rgb_dec_sample.shape)
    print("Decoded RGB sample dtype:", rgb_dec_sample.dtype)
    print("Decoded RGB sample device:", rgb_dec_sample.device)
    print("Decoded RGB sample min:", rgb_dec_sample.min())
    print("Decoded RGB sample max:", rgb_dec_sample.max())
    save_image(rgb_dec_sample.squeeze(0), os.path.join(out_path, "rgb_dec_sample.jpg"))
    
    
    
