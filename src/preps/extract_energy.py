import os
import traceback
import tqdm
import numpy as np
import torch

from src.data.components.audio_utils import AudioUtils
from src.preps.extract_base import BaseExtractor

class ExtractEnergy(BaseExtractor):
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
                wav = wav.squeeze(0)

                # split wav into 64ms frames with 20ms overlap
                frames = AudioUtils.frame(wav, frame_length=1024, hop_length=320)
                
                # extract frame-level energy
                energy = (frames ** 2).sum(axis=0).unsqueeze(0)
                save_path = fpath.replace(".wav", ".energy.pt")
                torch.save(energy, save_path)
                
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {fpath}: {e}")
                with open("error.log", "a") as f:
                    f.write(f"[ExtractEnergy] {fpath}: {e}\n")