import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = mel(
            sr=sampling_rate, 
            n_fft=n_fft, 
            n_mels=n_mel_channels, 
            fmin=mel_fmin, 
            fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        magnitude = torch.abs(fft)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        
        return log_mel_spec
    
class ReconstructionLoss(nn.Module):
    def __init__(self, 
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024,
                 sampling_rate=16000,
                 n_mel_channels=80,
                 mel_fmin=0.0,
                 mel_fmax=None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fft = Audio2Mel(n_fft=n_fft,
                             hop_length=hop_length,
                             win_length=win_length,
                             sampling_rate=sampling_rate,
                             n_mel_channels=n_mel_channels,
                             mel_fmin=mel_fmin,
                             mel_fmax=mel_fmax)
        
    def forward(self, x, G_x):
        S_x = self.fft(x)
        S_G_x = self.fft(G_x)
        
        loss = F.l1_loss(S_x, S_G_x)
        return loss