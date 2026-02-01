# dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Speech Commands v0.02 structure: folders per class + _background_noise_
        self.classes = sorted([d for d in os.listdir(self.root_dir) if not d.startswith('_')])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.audio_files = []
        self.labels = []
        
        # Assume split files like train/testing_list.txt exist or use folder logic
        # For simplicity: load all except background for now
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for wav_file in cls_dir.glob('*.wav'):
                self.audio_files.append(wav_file)
                self.labels.append(self.class_to_idx[cls])
        
        # In real notebook: filter by split using txt files
        print(f"Loaded {len(self.audio_files)} samples for {split}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        wav_path = self.audio_files[idx]
        label = self.labels[idx]
        
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        
        if waveform.shape[1] == 0:
            waveform = torch.zeros(1, 16000)
        
        if self.transform:
            spec = self.transform(waveform)
        else:
            spec = waveform  # fallback
        
        return spec.squeeze(0), label  # remove channel dim if mono
