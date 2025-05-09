#!/usr/bin/env python3
import argparse, subprocess, tempfile
from pathlib import Path
import torch, torchaudio
from transformers import AutoFeatureExtractor, MimiModel

# ---------- utils ---------------------------------------------------
def ensure_backend():
    if torchaudio.list_audio_backends():
        return
    raise RuntimeError("Error")
    

def extract_segment(video: Path, start: float, dur: float, sr: int) -> torch.Tensor:
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.check_call([
        "ffmpeg","-y","-ss",str(start),"-t",str(dur),"-i",str(video),
        "-ac","1","-ar",str(sr),"-c:a","pcm_s16le",str(tmp),"-loglevel","error"])
    wav, _ = torchaudio.load(tmp); tmp.unlink(missing_ok=True)
    return wav

def safe_load_wav(wav_path: Path, target_sr: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(wav_path)
    except Exception:
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        subprocess.check_call([
            "ffmpeg","-y","-i",str(wav_path),"-ac","1","-ar",str(target_sr),
            "-c:a","pcm_s16le",str(tmp),"-loglevel","error"])
        wav, sr = torchaudio.load(tmp); tmp.unlink(missing_ok=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

def pad_or_trim(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = wav.shape[-1]
    return (torch.nn.functional.pad(wav, (0, target_len - cur))
            if cur < target_len else wav[..., :target_len])

# ---------- main pipeline -------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("input"); p.add_argument("--start",type=float,default=0.0) 
    p.add_argument("--duration",type=float,default=3.0) # video time
    p.add_argument("--out_dir",default=".")
    args = p.parse_args()

    ensure_backend()

    IN, OUT = Path(args.input), Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    SR, SEC, N = 24_000, args.duration, int(24_000*args.duration) #We need 24OOO Hz for Mimi

    # 1) read
    if IN.suffix.lower()==".wav":
        wav = pad_or_trim(safe_load_wav(IN, SR), N)
    else:
        wav = extract_segment(IN, args.start, SEC, SR)
    print(f"Waveform ready : {wav.shape} échantillons @ {SR} Hz")

    # 2) encode Mimi
    ext = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    model = MimiModel.from_pretrained("kyutai/mimi")
    feats = ext(raw_audio=wav.squeeze(0), sampling_rate=SR, return_tensors="pt")
    codes = model.encode(feats["input_values"], feats["padding_mask"]).audio_codes
    torch.save(codes, OUT / "codes.pt")

    # 3) decode Mimi
    recon = model.decode(codes, feats["padding_mask"]).audio_values
    if recon.dim()==3: recon = recon.squeeze(0)       
    else:             recon = recon.unsqueeze(1).squeeze(0)  

    recon = recon.detach().cpu()                      
    torchaudio.save(str(OUT / "recon.wav"), recon, SR)
    print("✅ recon.wav et codes.pt saved in", OUT)

if __name__ == "__main__":
    main()
