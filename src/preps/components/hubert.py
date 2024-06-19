import os
import torch
import torchaudio


class Hubert:
    def __init__(
        self,
        device: str = "cuda",
    ):
        self.device = device

    def _load_model(self):
        print(f"Loading hubert checkpoint")
        hubert = torch.hub.load(
            "bshall/hubert:main",
            f"hubert_discrete",
            trust_repo=True,
        ).to(self.device)
        hubert.eval()
        self.model = hubert

    def extract_features(self, wav: torch.Tensor):
        if not hasattr(self, "model"):
            self._load_model()
        
        with torch.inference_mode():
            features = self.model.units(wav)
        return features


if __name__ == "__main__":
    from src.data.components.audio_utils import AudioUtils

    model = Hubert(device="cpu")
    wav, sr = AudioUtils.load_audio("data/sample.wav", sample_rate=16000)
    wav = AudioUtils.to_mono(wav)
    x = model.extract_features(wav.unsqueeze(0))
    print(x.shape, wav.shape[-1] // 320)
