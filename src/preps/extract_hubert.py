import os, glob

import torch
from src.preps.components.hubert import Hubert
from src.data.components.audio_utils import AudioUtils

from functools import partial
from multiprocessing import Process

import tqdm


class ExtractHubert:
    def __init__(self, filelist: list, device: str = "cpu"):
        self.filelist = filelist
        self.device = device

        self.model = None

    def _load_model(self):
        self.hubert = Hubert(device=self.device)

    def _run(self, rank):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        self._load_model()

        for fpath in tqdm.tqdm(self.filelist):
            wav, sr = AudioUtils.load_audio(fpath, sample_rate=16000)
            wav = AudioUtils.to_mono(wav)
            wav = wav.to(self.device)

            # extract features
            x = self.hubert.extract_features(wav, output_layer=7)

            # save extracted features
            save_path = fpath.replace(".wav", ".hubert.pt")
            torch.save(x, save_path)

    def run(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            processes = []
            for rank in range(num_gpus):
                p = Process(target=partial(self._run, rank))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            self._run(0)


if __name__ == "__main__":
    device = "cpu"
    filelist = ["data/sample.wav"]
    extractor = ExtractHubert(filelist, device)
    extractor.run()
