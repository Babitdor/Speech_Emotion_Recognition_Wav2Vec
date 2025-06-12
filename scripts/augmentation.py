import random
import torch
import numpy as np
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Gain

# Define emotion-specific augmentation pipelines
general_augmentation = Compose(
    [
        AddGaussianNoise(min_amplitude=0.003, max_amplitude=0.02, p=0.7),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.7),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
        Gain(min_gain_db=-12, max_gain_db=9, p=0.5),
    ]
)


def apply_augmentation(waveform, sample_rate):
    """
    waveform: torch.Tensor or np.ndarray, shape [channels, samples] or [samples]
    sample_rate: int
    """
    # Convert to numpy for audiomentations
    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.cpu().numpy()
    else:
        waveform_np = waveform

    # audiomentations expects shape [samples,] or [channels, samples]
    if waveform_np.ndim == 1:
        waveform_np = np.expand_dims(waveform_np, axis=0)

    augmented = general_augmentation(samples=waveform_np, sample_rate=sample_rate)

    # Return as torch.Tensor
    return torch.tensor(augmented, dtype=torch.float32)
