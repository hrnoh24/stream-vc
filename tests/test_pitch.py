import typing as T
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt

def estimate(
    signal: T.Union[T.List, np.ndarray, torch.Tensor],
    sample_rate: float,
    pitch_min: float = 80,
    pitch_max: float = 600,
    frame_stride: float = 0.02,
    threshold: float = 0.1,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """estimate the pitch (fundamental frequency) of a signal

    This function attempts to determine the pitch of a signal via the
    Yin algorithm. Accuracy can be improved by sampling the signal at a
    higher rate, especially for higher-frequency pitches, and by narrowing
    the values of pitch_min and pitch_max. For example, good values for
    speech signals are pitch_min=60 and pitch_max=500. frame_stride can also
    be tuned to the expected minimum rate of pitch change of the signal:
    10ms is commonly used for speech.

    The speed and memory usage of the algorithm are also determined by the
    pitch_min parameter, which is used to window the audio signal into
    2*sample_rate/pitch_min sliding windows. A higher pitch_min corresponds to
    less memory usage and faster running time.

    Args:
        signal: the signal vector (1D) or [batch, time] tensor to analyze
        sample_rate: sample rate, in Hz, of the signal
        pitch_min: expected lower bound of the pitch
        pitch_max: expected upper bound of the pitch
        frame_stride: overlapping window stride, in seconds, which determines
            the number of pitch values returned
        threshold: harmonic threshold value (see paper)

    Returns:
        pitch: PyTorch tensor of pitch estimations, one for each frame of
            the windowed signal, an entry of 0 corresponds to a non-periodic
            frame, where no periodic signal was detected
        cmnd_value: Cumulative Mean Normalized Difference Value at the estimated period
        unvoiced_predicate: Estimated Unvoiced (Aperiodic) Signal Predicate, 1 if unvoiced, 0 if voiced

    """

    signal = torch.as_tensor(signal)

    # convert frequencies to samples, ensure windows can fit 2 whole periods
    tau_min = int(sample_rate / pitch_max)
    tau_max = int(sample_rate / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = int(frame_stride * sample_rate)

    # compute the fundamental periods
    frames = _frame(signal, frame_length, frame_stride)
    cmdf = _diff(frames, tau_max)[..., tau_min:]
    tau = _search(cmdf, tau_max, threshold)

    # calculate the CMND value at the estimated period
    cmnd_value = cmdf[torch.arange(cmdf.size(0)), tau]
    
    # determine unvoiced predicate based on the threshold
    unvoiced_predicate = (cmnd_value >= threshold).int()

    # convert the periods to frequencies (if periodic) and output
    pitch = torch.where(
        tau > 0,
        sample_rate / (tau + tau_min + 1).type(signal.dtype),
        torch.tensor(0, device=tau.device).type(signal.dtype),
    )
    
    return pitch, cmnd_value, unvoiced_predicate


def _frame(signal: torch.Tensor, frame_length: int, frame_stride: int) -> torch.Tensor:
    # window the signal into overlapping frames, padding to at least 1 frame
    if signal.shape[-1] < frame_length:
        signal = torch.nn.functional.pad(signal, [0, frame_length - signal.shape[-1]])
    return signal.unfold(dimension=-1, size=frame_length, step=frame_stride)


def _diff(frames: torch.Tensor, tau_max: int) -> torch.Tensor:
    # compute the frame-wise autocorrelation using the FFT
    fft_size = 2 ** (-int(-np.log(frames.shape[-1]) // np.log(2)) + 1)
    fft = torch.fft.rfft(frames, fft_size, dim=-1)
    corr = torch.fft.irfft(fft * fft.conj())[..., :tau_max]

    # difference function (equation 6)
    sqrcs = torch.nn.functional.pad((frames * frames).cumsum(-1), [1, 0])
    corr_0 = sqrcs[..., -1:]
    corr_tau = sqrcs.flip(-1)[..., :tau_max] - sqrcs[..., :tau_max]
    diff = corr_0 + corr_tau - 2 * corr

    # cumulative mean normalized difference function (equation 8)
    return (
        diff[..., 1:]
        * torch.arange(1, diff.shape[-1], device=diff.device)
        / torch.maximum(
            diff[..., 1:].cumsum(-1),
            torch.tensor(1e-5, device=diff.device),
        )
    )


def _search(cmdf: torch.Tensor, tau_max: int, threshold: float) -> torch.Tensor:
    # mask all periods after the first cmdf below the threshold
    # if none are below threshold (argmax=0), this is a non-periodic frame
    first_below = (cmdf < threshold).int().argmax(-1, keepdim=True)
    first_below = torch.where(first_below > 0, first_below, tau_max)
    beyond_threshold = torch.arange(cmdf.shape[-1], device=cmdf.device) >= first_below

    # mask all periods with upward sloping cmdf to find the local minimum
    increasing_slope = torch.nn.functional.pad(cmdf.diff() >= 0.0, [0, 1], value=1)

    # find the first period satisfying both constraints
    return (beyond_threshold & increasing_slope).int().argmax(-1)


def plot_features(signal, sr, pitch, cmnd_value, unvoiced_predicate, hop_length):
    plt.figure(figsize=(15, 10))

    # 시간 벡터 생성
    times = np.arange(pitch.shape[0]) * hop_length / sr

    # 추정된 피치 시각화
    plt.subplot(3, 1, 1)
    plt.plot(times, pitch.numpy(), label='Estimated f0', color='yellow', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Estimated Pitch (f0)')
    plt.legend()

    # CMND 값 시각화
    plt.subplot(3, 1, 2)
    plt.plot(times, cmnd_value.numpy(), label='CMND Value', color='green', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('CMND Value')
    plt.title('Cumulative Mean Normalized Difference Value')
    plt.legend()

    # Unvoiced Predicate 시각화
    plt.subplot(3, 1, 3)
    plt.plot(times, unvoiced_predicate.numpy(), label='Unvoiced Predicate', color='blue', linewidth=2, linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel('Unvoiced (0 or 1)')
    plt.title('Unvoiced Predicate')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 예제 사용법
if __name__ == "__main__":
    # 샘플링 레이트 16000으로 설정
    sr = 16000
    hop_length = 256

    # 예제 오디오 신호 로드 및 리샘플링
    signal, _ = librosa.load(librosa.ex('trumpet'), sr=sr)
    
    # 특징 추출
    pitch, cmnd_value, unvoiced_predicate = estimate(signal, sr, frame_stride=hop_length / sr, threshold=0.15)

    # 시각화
    plot_features(signal, sr, pitch, cmnd_value, unvoiced_predicate, hop_length)
