import os
import torch
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from huggingface_hub import snapshot_download
from torchvision import transforms as T

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
                Cosmos-0.1-Tokenizer-DV4x8x8, (Shape out: (B, C, H/4, W/4))
                Cosmos-0.1-Tokenizer-DV8x8x8,
                Cosmos-0.1-Tokenizer-DV8x16x16,
            device (torch.device): Device on which the computation will occur.
        """
        self.model_name = model_name
        self.type = "continuous" if "CV" in model_name else "discrete" if "DV" in model_name else None
        if self.type is None:
            raise ValueError(f"Invalid model name: {model_name}. Must contain 'CV' or 'DV'.")

        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        encoder_path = os.path.join(self.cache_dir, model_name, 'encoder.jit')
        decoder_path = os.path.join(self.cache_dir, model_name, 'decoder.jit')
        snapshot_download(repo_id=f"nvidia/{model_name}", local_dir=os.path.join(cache_dir, model_name))
        self.tokenizer = CausalVideoTokenizer(checkpoint_enc=encoder_path,
                                              checkpoint_dec=decoder_path,
                                              device=self.device,
                                              dtype = "bfloat16")

        self.normalize_cosmos = T.Normalize(mean = 0.5,
                                            std = 0.5)
        
        self.denormalize_cosmos = T.Normalize(mean = -1.0,
                                              std = 2.0) 


    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        # input_tensor should be  of shape : [B, C, T, H, W]
        """
        video = self.normalize_cosmos(video)
        input_tensor = video.to(self.device).to(torch.bfloat16)
        (tokens, _) = self.tokenizer.encode(input_tensor)  
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        output_tensor = self.tokenizer.decode(tokens).clamp(-1,1).to(torch.float32)
        return self.denormalize_cosmos(output_tensor)
        

if __name__ == "__main__":
    from dataset import MyImageDataset
    from nanofm.data.utils import save_video


    out_path = ".local_cache/video_tokenizer"
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dataset = MyImageDataset(data_path="/work/com-304/SAGA/raw",
                                    csv_file="/home/bousquie/COM-304-FM/SAGA_COM-304/.local_cache/small_vgg.csv",
                                    device=device)
    video_tok = VideoTokenizer(model_name="Cosmos-0.1-Tokenizer-DV8x8x8", device=device)

    # Test RGB
    # ==== Choose an image
    frames_sample = img_dataset[5]["frames"]
    os.makedirs(out_path, exist_ok=True)
    save_video(frames_sample, os.path.join(out_path, "video.gif"))
    # ==== Print information
    print("Frames sample shape:", frames_sample.shape)
    print("Frames sample dtype:", frames_sample.dtype)
    print("Frames sample device:", frames_sample.device)
    print("Frames sample min:", frames_sample.min())
    print("Frames sample max:", frames_sample.max())

    # ==== Encode
    tokens = video_tok.encode(frames_sample.unsqueeze(0))
    print("Tokens shape:", tokens.shape)
    print("Tokens dtype:", tokens.dtype)
    print("Tokens device:", tokens.device)
    print("Tokens min:", tokens.min())
    print("Tokens max:", tokens.max())

    # ==== Decode
    vid_dec_sample = video_tok.decode(tokens)
    vid_dec_sample = vid_dec_sample.squeeze(0)
    print("Decoded RGB sample shape:", vid_dec_sample.shape)
    print("Decoded RGB sample dtype:", vid_dec_sample.dtype)
    print("Decoded RGB sample device:", vid_dec_sample.device)
    print("Decoded RGB sample min:", vid_dec_sample.min())
    print("Decoded RGB sample max:", vid_dec_sample.max())
    save_video(vid_dec_sample, os.path.join(out_path, "decod_video.gif"))
        

