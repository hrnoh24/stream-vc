import torch
from torchaudio.transforms import MelSpectrogram


def spectral_reconstruction_loss(x, G_x, sr=16000, eps=1e-4, device="cpu"):
    L = 0
    for i in range(6, 12):
        s = 2**i
        alpha_s = (s / 2) ** 0.5
        melspec = MelSpectrogram(
            sample_rate=sr,
            n_fft=s,
            hop_length=s // 4,
            n_mels=8,
            wkwargs={"device": device},
        ).to(device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)

        loss = (S_x - S_G_x).abs().sum() + alpha_s * (
            ((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps)) ** 2).sum(
                dim=-2
            )
            ** 0.5
        ).sum()
        L += loss

    return L
