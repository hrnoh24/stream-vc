import os
import torch
import fairseq

from src.utils.utils import download


class Hubert:
    def __init__(
        self,
        ckpt_path: str = "pretrained_models/hubert/hubert_base_ls960.pt",
        device: str = "cuda",
    ):
        self.ckpt_path = ckpt_path
        self.device = device
        if not os.path.exists(ckpt_path):
            model_name = os.path.basename(ckpt_path)
            dir_name = os.path.dirname(ckpt_path)
            os.makedirs(dir_name, exist_ok=True)

            download(
                f"https://dl.fbaipublicfiles.com/hubert/{model_name}",
                ckpt_path,
            )

    def _load_model(self):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.ckpt_path]
        )
        model = models[0].eval()
        model.to(self.device)
        self.model = model

    def extract_features(self, wav: torch.Tensor, output_layer=7):
        if not hasattr(self, "model"):
            self._load_model()
        x, _ = self.model.extract_features(
            source=wav, padding_mask=None, mask=False, output_layer=output_layer
        )
        return x


if __name__ == "__main__":
    from src.data.components.audio_utils import AudioUtils

    model = Hubert("dummy", device="cpu")
    wav, sr = AudioUtils.load_audio("data/sample.wav", sample_rate=16000)
    wav = AudioUtils.to_mono(wav)
    x = model.extract_features(wav, output_layer=7)
    print(x.shape, wav.shape[-1] // 320)
