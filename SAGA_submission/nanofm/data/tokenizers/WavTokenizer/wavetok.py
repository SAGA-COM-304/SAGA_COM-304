import os
import torch
import torchaudio
from .encoder.utils import convert_audio
from .decoder.pretrained import WavTokenizer
import glob


class WavAudioTokenizer:
    """
    Wrapper pour WavTokenizer (40 tokens/s).

    Args:
        config_path (str): Chemin vers le fichier YAML de configuration (ex: "./configs/small-600-24k.yaml").
        checkpoint_path (str): Chemin vers le checkpoint .ckpt (ex: "./checkpoints/wavtokenizer-small-600-24k-4096.ckpt").
        device (torch.device, optional): périphérique d'exécution. Par défaut GPU si dispo, sinon CPU.
    """
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device = None
    ):
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Sampling rate = 24 kHz
        self.sr = 24_000

        # Instanciation du tokenizer
        # Note : from_pretrained0802 prend en entrée le config YAML et le checkpoint
        self.tokenizer = WavTokenizer.from_pretrained0802(
            config_path, 
            checkpoint_path
        ).to(self.device)

    def encode(self, wav: torch.Tensor, sr: int) -> torch.LongTensor:
        """
        Encode un tenseur audio en codes discrets.

        Args:
            wav (torch.Tensor): Tensor shape [1, T] ou [C, T] (mono recommandé).
            sr (int): sample rate d'entrée.

        Returns:
            codes (torch.LongTensor): tenseur [n_q, T_tokens] de codes discrets.
        """
        # Normaliser et resampler
        wav = convert_audio(wav, sr, self.sr, 1)
        wav = wav.to(self.device)

        # bandwidth_id = 0 correspond à la version "small" (40 tokens/s)
        bandwidth_id = torch.tensor([0], device=self.device)

        # encode_infer renvoie (features, discrete_code)
        _, codes = self.tokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)  
        return codes

    def decode(self, codes: torch.LongTensor) -> torch.Tensor:
        """
        Reconstruit l'audio depuis les codes discrets.

        Args:
            codes (torch.LongTensor): tenseur [n_q, T_tokens] de codes discrets.

        Returns:
            audio (torch.Tensor): tenseur audio reconstruite, shape [1, T_wav].
        """
        # Convertir codes en features
        features = self.tokenizer.codes_to_features(codes)
        bandwidth_id = torch.tensor([0], device=self.device)

        # Décodage
        audio = self.tokenizer.decode(features, bandwidth_id=bandwidth_id)
        return audio

# Exemple d'utilisation :
if __name__ == "__main__":
    folder = "/work/com-304/SAGA/raw/audios"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chemins vers config & checkpoint (à adapter)
    #config = "/home/godey/SAGA_COM-304/nanofm/data/tokenizers/WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    #ckpt   = "/work/com-304/SAGA/wavtok/wavtokenizer_small_320_24k_4096.ckpt"

    config = '/home/godey/SAGA_COM-304/nanofm/data/tokenizers/WavTokenizer/configs/mine_wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
    ckpt = '/work/com-304/SAGA/wavtok/train/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn/lightning_logs/version_22/checkpoints/last.ckpt'

    # Chargement
    tok = WavAudioTokenizer(config, ckpt)

    # Exemple de test unique
    for i in range(10):
        sample_path = glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)[10 + i]
        wav, sr = torchaudio.load(sample_path)

        # Chargement d'un fichier audio
        wav, sr = torchaudio.load(sample_path)   
        torchaudio.save(f"data_gpt2_tokenizer/original{i}.wav", wav.cpu(), sample_rate=24000)

        # Encodage
        codes = tok.encode(wav, sr)
        print("Codes shape:", codes.shape)
        print('CODES MAX AND MIN', codes.max(), codes.min())

        # Décodage
        wav_rec = tok.decode(codes)
        torchaudio.save(f"data_gpt2_tokenizer/reconstructed{i}.wav", wav_rec.cpu(), sample_rate=24000)
