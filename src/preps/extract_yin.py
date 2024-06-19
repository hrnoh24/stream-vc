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
        
    def _normalize_pitch(self, pitch):
        # pitch = pitch.numpy()
        nonzero = pitch > 0
        if len(nonzero) == 0:
            return pitch
        
        mean = torch.mean(pitch[nonzero])
        std = torch.std(pitch[nonzero])
        pitch = (pitch - mean) / std
        
        pitch_norm = torch.zeros_like(pitch)
        pitch_norm[nonzero] = pitch[nonzero]
        return pitch_norm
        
    def _run(self, rank, filelist):
        for fpath in tqdm.tqdm(filelist):
            try:
                wav, sr = AudioUtils.load_audio(fpath, sample_rate=16000)
                wav = AudioUtils.to_mono(wav)
                wav = wav.squeeze(0)

                # extract features
                pitches = []
                cmdf_vs = []
                cmdf_uvs = []
                for threshold in [0.05, 0.1, 0.15]:
                    pitch, cmdf_v, cmdf_uv = estimate(wav.numpy(), sr, threshold=threshold)
                    pitch_norm = self._normalize_pitch(pitch)
                    
                    pitches.append(pitch_norm[np.newaxis, :])
                    cmdf_vs.append(cmdf_v[np.newaxis, :])
                    cmdf_uvs.append(cmdf_uv[np.newaxis, :])
                
                pitches = np.concatenate(pitches, axis=0)
                cmdf_vs = np.concatenate(cmdf_vs, axis=0)
                cmdf_uvs = np.concatenate(cmdf_uvs, axis=0)
                
                features = np.concatenate([pitches, cmdf_vs, cmdf_uvs], axis=0)
                features = torch.FloatTensor(features)
                save_path = fpath.replace(".wav", ".pitch.pt")
                torch.save(features, save_path)
                
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {fpath}: {e}")
                with open("error.log", "a") as f:
                    f.write(f"[ExtractPitch] {fpath}: {e}\n")