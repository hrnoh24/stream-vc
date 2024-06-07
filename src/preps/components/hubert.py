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
        bundle = torchaudio.pipelines.HUBERT_BASE
        model = bundle.get_model()
        model.eval()
        model.to(self.device)
        self.model = model

    def extract_features(self, wav: torch.Tensor, output_layer=7):
        if not hasattr(self, "model"):
            self._load_model()
        features, _ = self.model.extract_features(wav, num_layers=output_layer)
        return features[-1]


if __name__ == "__main__":
    from src.data.components.audio_utils import AudioUtils

    model = Hubert(device="cpu")
    wav, sr = AudioUtils.load_audio("data/sample.wav", sample_rate=16000)
    wav = AudioUtils.to_mono(wav)
    x = model.extract_features(wav, output_layer=7)
    print(x.shape, wav.shape[-1] // 320)
