import torch
import torchaudio
from pathlib import Path
from typing import Union, Optional, Tuple
from transformers import AutoFeatureExtractor, MimiModel

class AudioTokenizer:
    """Audio Tokenizer using Mimi model - Always returns padding mask for optimal decoding."""
    
    def __init__(self, device: str = "cpu"):
        self.sr = 24_000
        self.device = torch.device(device)
        
        # Load model and extractor
        self.extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        self.model = MimiModel.from_pretrained("kyutai/mimi").to(self.device)

    def encode(
        self, 
        audio: Union[str, Path, torch.Tensor],
        num_quantizers: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to discrete codes.
        
        Always returns both codes and padding mask for optimal decoding.
        
        Returns:
            Tuple of (codes, padding_mask)
        """
        if isinstance(audio, (str, Path)):
            wav = self._load_audio(Path(audio))
        else:
            wav = audio
            
        wav = wav.to(self.device)

        with torch.no_grad():
            inputs = self.extractor(
                raw_audio=wav.squeeze(0).cpu().numpy(),
                sampling_rate=self.extractor.sampling_rate,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if num_quantizers is not None:
                encoder_outputs = self.model.encode(
                    inputs["input_values"], 
                    inputs["padding_mask"],
                    num_quantizers=num_quantizers
                )
            else:
                encoder_outputs = self.model.encode(
                    inputs["input_values"], 
                    inputs["padding_mask"]
                )
            
            codes = encoder_outputs.audio_codes.squeeze(0)
            
        return codes.detach().cpu(), inputs["padding_mask"].detach().cpu()

    def decode(self, codes: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode audio codes back to waveform"""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
            
        codes = codes.to(self.device)
        
        # Ensure padding mask has correct dimensions if provided
        if padding_mask is not None:
            if padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(0)
            padding_mask = padding_mask.to(self.device)
        
        with torch.no_grad():
            # Use padding mask if provided for better reconstruction
            if padding_mask is not None:
                decoder_outputs = self.model.decode(codes, padding_mask)
            else:
                decoder_outputs = self.model.decode(codes)
            
            if isinstance(decoder_outputs, (tuple, list)):
                audio_values = decoder_outputs[0]
            else:
                audio_values = decoder_outputs.audio_values
                
            # Ensure correct output shape (1, T)
            if audio_values.dim() == 3:
                audio_values = audio_values.squeeze(0)
            if audio_values.dim() == 1:
                audio_values = audio_values.unsqueeze(0)
                
        return audio_values.detach().cpu()

    def get_model_info(self) -> dict:
        """Get model information"""
        config = self.model.config
        return {
            "num_quantizers": getattr(config, "num_quantizers", "Unknown"),
            "codebook_size": getattr(config, "codebook_size", "Unknown"),
            "sampling_rate": config.sampling_rate,
            "frame_rate": getattr(config, "frame_rate", "Unknown"),
        }