import subprocess, tempfile, argparse
from pathlib import Path
from typing import Union, Optional

import torch, torchaudio
from transformers import AutoFeatureExtractor, MimiModel

# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------
def _ensure_backend() -> None:
    if not torchaudio.list_audio_backends():
        raise RuntimeError(
            "torchaudio didn't find any audio backend.\n"
        )

def _ffmpeg_extract(inp: Path, out: Path, start: float, dur: float, sr: int) -> None:
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-t", str(dur),
           "-i", str(inp), "-ac", "1", "-ar", str(sr),
           "-c:a", "pcm_s16le", str(out), "-loglevel", "error"]
    subprocess.check_call(cmd)

# --------------------------------------------------------------------
# main class
# --------------------------------------------------------------------
class AudioTokenizer:
    def __init__(self, target_sr: int = 24_000, device: str = "cpu"):
        """
        target_sr : 24 OOO Hz (asked by Mimi)
        device    : cpu, cuda (scitas), mps (mac)
        """
        _ensure_backend()
        self.sr = target_sr
        self.device = torch.device(device)

        # ↓ downloads only once
        self.extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi") # normalizes .wav in tensor
        self.model     = MimiModel.from_pretrained("kyutai/mimi").to(self.device)

    # --------------- I/O WAV ----------------------------------------
    def _load_any_audio(self, path: Path, start: float, dur: float) -> torch.Tensor:
        """loads a video or a .Wav and returns a mono tensor (1,N) on CPU of 24 kHz."""

        if path.suffix.lower() == ".wav": # if a .wav file
            try:
                wav, sr = torchaudio.load(path) #tries to load with torchaudio
            except Exception:  # if it doesn't work it converts it with ffmpeg (WAV PCM + propre)
                tmp = Path(tempfile.mktemp(suffix=".wav"))
                _ffmpeg_extract(path, tmp, start, dur, self.sr)
                wav, sr = torchaudio.load(tmp)
                tmp.unlink(missing_ok=True)
        else:  # if a video file ffmpeg the audio extract from the video
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            _ffmpeg_extract(path, tmp, start, dur, self.sr)
            wav, sr = torchaudio.load(tmp); tmp.unlink(missing_ok=True) # converts it to .wav

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav

    # --------------- ENCODE ----------------------------------------
    def encode(
        self,
        audio: Union[str, Path, torch.Tensor],
        start: float = 0.0,
        dur: Optional[float] = None,
    ) -> torch.Tensor:
        """
        audio : path (str/Path) or tensor (1, N) already at 24 kHz.
        start/dur : start time of the video / duration
        Returns a tensor int64 (frames, 32).
        """
        if isinstance(audio, (str, Path)):
            wav = self._load_any_audio(Path(audio), start, dur or 1e9)
        else:  # tensor déjà fourni
            wav = audio
        wav = wav.to(self.device)

        feats  = self.extractor(raw_audio=wav.squeeze(0),
                                sampling_rate=self.sr,
                                return_tensors="pt").to(self.device)

        codes  = self.model.encode(
                    feats["input_values"], feats["padding_mask"]
                 ).audio_codes.squeeze(0)         # (frames, 32)
        return codes.detach().cpu()

    # --------------- DECODE ----------------------------------------
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes : tensor (frames, 32) or (B, frames, 32). B -> batchsize
        Returns a wav tensor (1, T) at 24 kHz (on CPU, detached).
        """
        if codes.dim() == 2:   # (frames,32) → (1,frames,32)
            codes = codes.unsqueeze(0)
        out = self.model.decode(codes.to(self.device)).audio_values
        wav = out.squeeze(0) if out.dim() == 3 else out
        return wav.detach().cpu()
    
    