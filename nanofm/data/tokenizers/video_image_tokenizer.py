import os
import torch
from cosmos_tokenizer.image_lib import ImageTokenizer as BaseImageTokenizer
from huggingface_hub import snapshot_download
from torchvision import transforms as T
from einops import rearrange

class VideoImageTokenizer:
    def __init__(self,
            model_name: str,
            device: torch.device,
            cache_dir: str = ".local_cache",
    ):
        """
        Initializes the VideoTokenizer with the specified model name and device.

        Arguments:
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
                                              dtype = "bfloat16")

        self.normalize_cosmos = T.Normalize(mean = 0.5,
                                            std = 0.5)
        
        self.denormalize_cosmos = T.Normalize(mean = -1.0,
                                              std = 2.0) 


    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        # input_tensor should be  of shape : B, C, T, H, W -> BT C H W -token> BT 16, 16 -> B T 16, 16
        """

        b, c, t, h, w = video.shape
        video = self.normalize_cosmos(video)
        video = rearrange(video, "b c t h w -> (b t) c h w")
        input_tensor = video.to(self.device).to(torch.bfloat16)
        tokens, _ = self.tokenizer.encode(input_tensor)
        tokens = rearrange(tokens, "(b t) h w -> b t h w",b=b, t=t)
        return tokens
        
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """"
        # input_tensor should be  of shape : [B T 16 16] ---> BT 16 16 ---decode---> BT C H W ---> B C T H W 
        """
        b, t, _, _ = tokens.shape
        tokens = rearrange(tokens, "b t h w -> (b t) h w")
        output_tensor = self.tokenizer.decode(tokens).clamp(-1, 1).to(torch.float32)
        un_normalized = self.denormalize_cosmos(output_tensor)
        video = rearrange(un_normalized, "(b t) c h w -> b c t h w", b=b, t=t)
        return video

if __name__ == "__main__":
    from dataset import MyImageDataset
    from nanofm.data.utils import save_video


    out_path = ".local_cache/video_image_tokenizer"
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dataset = MyImageDataset(data_path="/work/com-304/SAGA/raw",
                                    csv_file="/home/bousquie/COM-304-FM/SAGA_COM-304/.local_cache/small_vgg.csv",
                                    device=device)
    video_tok = VideoImageTokenizer(model_name="Cosmos-0.1-Tokenizer-DI16x16", device=device)

    # Test RGB
    # ==== Choose an image
    frames_sample = torch.stack((img_dataset[5]["frames"], img_dataset[6]["frames"]))
    os.makedirs(out_path, exist_ok=True)
    save_video(frames_sample[0], os.path.join(out_path, "video1.gif"))
    save_video(frames_sample[1], os.path.join(out_path, "video2.gif"))
    # ==== Print information
    print("Frames sample shape:", frames_sample.shape)
    print("Frames sample dtype:", frames_sample.dtype) 
    print("Frames sample device:", frames_sample.device)
    print("Frames sample min:", frames_sample.min())
    print("Frames sample max:", frames_sample.max())

    # ==== Encode
    tokens = video_tok.encode(frames_sample)
    print("Tokens shape:", tokens.shape)
    print("Tokens dtype:", tokens.dtype)
    print("Tokens device:", tokens.device)
    print("Tokens min:", tokens.min())
    print("Tokens max:", tokens.max())

    # ==== Decode
    vid_dec_sample = video_tok.decode(tokens)
    print("Decoded RGB sample shape:", vid_dec_sample.shape)
    print("Decoded RGB sample dtype:", vid_dec_sample.dtype)
    print("Decoded RGB sample device:", vid_dec_sample.device)
    print("Decoded RGB sample min:", vid_dec_sample.min())
    print("Decoded RGB sample max:", vid_dec_sample.max())
    save_video(vid_dec_sample[0], os.path.join(out_path, "decod_video1.gif"))
    save_video(vid_dec_sample[1], os.path.join(out_path, "decod_video2.gif"))


        

