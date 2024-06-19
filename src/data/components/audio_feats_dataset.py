import torch
from torch.utils.data import Dataset
import os, glob

from src.data.components.audio_utils import AudioUtils

class AudioFeatsDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 ext=".wav",
                 filelist=None,
                 ) -> None:
        super().__init__()
        
        self.root_dir = root_dir
        self.ext = ext
        
        if not filelist:
            filelist = glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True)
        else:
            try:
                with open(filelist, "r") as f:
                    filelist = f.readlines()
                    filelist = [os.path.join(root_dir, f.strip()) for f in filelist]
            except Exception as e:
                print(f"Error reading filelist: {e}")
                exit(-1)
        self.filelist = filelist
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        fpath = self.filelist[idx]
        wav, sr = AudioUtils.load_audio(fpath, sample_rate=16000)
        wav = AudioUtils.to_mono(wav)
        
        pitch = torch.load(fpath.replace(self.ext, ".pitch.pt"))
        energy = torch.load(fpath.replace(self.ext, ".energy.pt"))
        hubert = torch.load(fpath.replace(self.ext, ".hubert.pt"))
        
        feature_len = hubert.shape[-1]
        # excerpt 75 frames
        if feature_len > 75:
            start = torch.randint(0, feature_len - 75, (1,)).item()
            pitch = pitch[:, start:start+75]
            energy = energy[:, start:start+75]
            hubert = hubert[start:start+75]
            wav = wav[:, start*320:(start+75)*320]
        else:
            pitch = torch.nn.functional.pad(pitch, (0, 75 - feature_len), "constant", 0)
            energy = torch.nn.functional.pad(energy, (0, 75 - feature_len), "constant", 0)
            hubert = torch.nn.functional.pad(hubert, (0, 75 - feature_len), "constant", 0)
            wav = torch.nn.functional.pad(wav, (0, (75 - feature_len) * 320), "constant", 0)
        
        return wav, pitch, energy, hubert
        