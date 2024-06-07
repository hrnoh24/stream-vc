import os
import traceback
import tqdm
import numpy as np
import torch

from src.data.components.audio_utils import AudioUtils
from src.preps.components.yin import estimate
from src.preps.extract_base import BaseExtractor

class ExtractYin(BaseExtractor):
    def __init__(self, 
                 root_dir: str, 
                 num_workers: int = 1,
                 device: str = "cpu"):
        super().__init__(root_dir, num_workers=num_workers, device=device)
        
    def _run(self, rank, filelist):
        for fpath in tqdm.tqdm(filelist):
            try:
                wav, sr = AudioUtils.load_audio(fpath, sample_rate=16000)
                wav = AudioUtils.to_mono(wav)

                # extract features
                features = []
                for threshold in [0.05, 0.1, 0.15]:
                    pitch, cmdf_v, cmdf_uv = estimate(wav.numpy(), sr, threshold=threshold)
                    features.append(pitch)
                    features.append(cmdf_v)
                    features.append(cmdf_uv)
                features = np.concatenate(features, axis=0)
                features = torch.FloatTensor(features)
                save_path = fpath.replace(".wav", ".pitch.pt")
                torch.save(features, save_path)
                
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {fpath}: {e}")
                with open("error.log", "a") as f:
                    f.write(f"[ExtractPitch] {fpath}: {e}\n")