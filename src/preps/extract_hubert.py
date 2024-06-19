import os, glob

import torch
from src.preps.components.hubert import Hubert
from src.data.components.audio_utils import AudioUtils

from src.preps.extract_base import BaseExtractor

import tqdm

class ExtractHubert(BaseExtractor):
    def __init__(self, 
                 root_dir: str, 
                 num_workers: int = 1,
                 device: str = "cpu"):
        super().__init__(root_dir, num_workers=num_workers, device=device)
        self.model = None

    def _load_model(self):
        self.hubert = Hubert(device=self.device)

    def _run(self, rank, filelist):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        self._load_model()

        for fpath in tqdm.tqdm(filelist):
            try:
                wav, sr = AudioUtils.load_audio(fpath, sample_rate=16000)
                wav = AudioUtils.to_mono(wav)
                wav = wav.unsqueeze(0).to(self.device)

                # extract features
                x = self.hubert.extract_features(wav)

                # save extracted features
                save_path = fpath.replace(".wav", ".hubert.pt")
                torch.save(x, save_path)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                with open("error.log", "a") as f:
                    f.write(f"[ExtractHubert] {fpath}: {e}\n")


if __name__ == "__main__":
    device = "cpu"
    filelist = ["data/sample.wav"]
    extractor = ExtractHubert(filelist, device)
    extractor.run()
