import torch
from transformers import MimiModel

class AudioTokenizer:
    """Audio Tokenizer using Mimi model - Always returns padding mask for optimal decoding."""
    
    def __init__(self, device = torch.device("cpu")):
        self.sr = 24_000
        self.num_quantizers = 32
        self.device = device
        
        # Load model and extractor
        self.model = MimiModel.from_pretrained("kyutai/mimi").to(self.device)

    def encode(
        self, 
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode audio to discrete codes.
        Always returns both codes and padding mask for optimal decoding.

        Inputs:
            audio = [Batch, SAMPLING_F * duration]
        Returns:
            Tuple of (codes, padding_mask)
        """ 

        audio = audio.unsqueeze(1).to(self.device) # [Batch, SAMPLING_F * duration] => [Batch, Channel = 1, Sampling_f * duration]

        with torch.no_grad():
            encoder_outputs = self.model.encode(audio, 
                                                num_quantizers= self.num_quantizers)
            codes = encoder_outputs.audio_codes
            
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode audio codes back to waveform
        Inputs : 
        codes = [Batch, codebook, frames] or [Batch]

        Return:
        audio = [batch, channels, time]

        """
        
        if codes.ndim == 2 : #Resize if output of model is of the form [Batch, num_quantizers * 63]
            B, _ = codes.shape
            codes = codes.reshape(B, self.num_quantizers, -1)
        
        with torch.no_grad():
            decoder_outputs = self.model.decode(codes)
            audio_values = decoder_outputs.audio_values
            audio_values.squeeze(1)

        return audio_values.detach()


if __name__ == "__main__":
    from dataset import MyImageDataset
    from nanofm.data.utils import save_audio
    import os
    

    out_path = ".local_cache/audio_tokenizer"
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dataset = MyImageDataset(data_path="/work/com-304/SAGA/raw",
                                    csv_file="/home/bousquie/COM-304-FM/SAGA_COM-304/.local_cache/small_vgg.csv",
                                    device=device)
    audio_tok = AudioTokenizer(device=device)

    # Test Audio
    # ==== Choose an audio
    audio_sample = img_dataset[2]["audios"]
    save_audio(audio_sample, os.path.join(out_path, "audio_sample"))
    # audio_sample2 = img_dataset[1]["audios"]
    
    # audio_sample = torch.stack((audio_sample, audio_sample2))
    
    # ==== Print information
    print("Audio sample shape:", audio_sample.shape) # 120_000
    print("Audio sample dtype:", audio_sample.dtype) 
    print("Audio sample device:", audio_sample.device) 
    print("Audio sample min:", audio_sample.min())
    print("Audio sample max:", audio_sample.max())

    # ==== Encode
    tokens = audio_tok.encode(audio_sample.unsqueeze(0))
    # tokens = audio_tok.encode(audio_sample)
    print("Tokens shape:", tokens.shape)
    print("Tokens dtype:", tokens.dtype)
    print("Tokens device:", tokens.device)
    print("Tokens min:", tokens.min())
    print("Tokens max:", tokens.max())

    # ==== Decode
    audio_dec_sample = audio_tok.decode(tokens)
    print("Decoded Audio sample shape:", audio_dec_sample.shape)
    print("Decoded Audio sample dtype:", audio_dec_sample.dtype)
    print("Decoded Audio sample device:", audio_dec_sample.device)
    print("Decoded Audio sample min:", audio_dec_sample.min())
    print("Decoded Audio sample max:", audio_dec_sample.max())

    save_audio(audio_dec_sample, os.path.join(out_path, "audio_dec_sample"))

        