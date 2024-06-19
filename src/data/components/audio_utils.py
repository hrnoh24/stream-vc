import numpy as np
import torch
import torchaudio
import librosa


class AudioUtils:
    @staticmethod
    def load_audio(fpath: str, sample_rate: int = None) -> torch.Tensor:
        audio, sr = torchaudio.load(fpath)
        if sample_rate is not None:
            if sr != sample_rate:
                audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
                sr = sample_rate

        return audio, sr

    @staticmethod
    def to_mono(audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio

    @staticmethod
    def random_excerpt(audio: torch.Tensor, duration: int) -> torch.Tensor:
        max_duration = audio.shape[1]
        if max_duration <= duration:
            return audio

        start = np.random.randint(0, max_duration - duration)
        return audio[:, start : start + duration]
    
    @staticmethod
    def frame(audio: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
        audio_np = audio.numpy()
        audio_np = np.pad(audio_np, (0, frame_length // 2), mode="constant")
        frames = librosa.util.frame(audio_np, frame_length=frame_length, hop_length=hop_length)
        return torch.tensor(frames)


if __name__ == "__main__":
    fpath = "data/sample.wav"
    audio, sr = AudioUtils.load_audio(fpath, sample_rate=16000)
    audio = AudioUtils.to_mono(audio)
    audio = AudioUtils.random_excerpt(audio, duration=3000)
    print(audio.shape, sr)
