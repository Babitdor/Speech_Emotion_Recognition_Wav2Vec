from torch.utils.data import Dataset, DataLoader
from scripts.augmentation import apply_augmentation
import torchaudio
import json
import torch
import random
import librosa
import numpy as np


class SERDataLoader(Dataset):
    def __init__(
        self,
        metadata_file,
        processor,
        transform=None,
        target_transform=None,
        label_map=None,
    ):
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        self.processor = processor
        self.label_map = (
            label_map
            if label_map
            else {
                "neutral": 0,
                "happy": 1,
                "sad": 2,
                "angry": 3,
            }
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = self.metadata[idx]["path"]
        emotion = self.metadata[idx]["emotion"]

        target_length_sec = 2
        target_sr = 16000
        target_legnth = target_length_sec * target_sr

        waveform, sample_rate = librosa.load(audio_path, sr=target_sr)
        waveform = waveform.squeeze()

        if len(waveform) < target_legnth:
            pad_width = target_legnth - len(waveform)
            waveform = np.pad(waveform, (0, pad_width), mode="constant")
        else:
            waveform = waveform[:target_legnth]

        waveform = torch.tensor(
            waveform,
            dtype=torch.float32,
        )

        if self.transform:
            waveform = apply_augmentation(waveform=waveform, sample_rate=sample_rate)

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()

        audio_inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_attention_mask=True,
            padding=False,
        )

        label_id = self.label_map[emotion]

        return {
            "input_values": audio_inputs["input_values"][0],
            "attention_mask": audio_inputs["attention_mask"][0],
            "labels": label_id,
        }
