import numpy as np
from scipy.signal import decimate


def zscore_per_sensor(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def downsample_signal(x: np.ndarray, factor: int = 4) -> np.ndarray:
    if factor <= 1:
        return x.astype(np.float32)

    return decimate(
        x,
        q=factor,
        axis=1,
        zero_phase=True,
    ).astype(np.float32)


def create_windows(
    x: np.ndarray,
    label: int,
    window_size: int,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray]:

    _, n_timepoints = x.shape

    windows = []

    for start in range(0, n_timepoints - window_size + 1, step_size):
        stop = start + window_size
        windows.append(x[:, start:stop])

    windows = np.stack(windows).astype(np.float32)
    labels = np.full(len(windows), label, dtype=np.int64)

    return windows, labels


def preprocess_recording(
    x: np.ndarray,
    label: int,
    original_sampling_rate: int = 2034,
    downsample_factor: int = 4,
    window_seconds: float = 2.0,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:

    x = zscore_per_sensor(x)
    x = downsample_signal(x, factor=downsample_factor)

    sampling_rate = original_sampling_rate / downsample_factor
    window_size = int(window_seconds * sampling_rate)
    step_size = int(window_size * (1 - overlap))

    return create_windows(
        x=x,
        label=label,
        window_size=window_size,
        step_size=step_size,
    )