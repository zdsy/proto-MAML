import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T


class ESC50Dataset(Dataset):
    def __init__(self, base_path, meta_data):
        self.base_path = base_path
        self.meta_data = meta_data
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            # Handle list of indices
            return [self.get_single_item(i) for i in idx]
        else:
            # Handle single index
            return self.get_single_item(idx)

    def get_single_item(self, idx):
        row = self.meta_data.iloc[idx]
        filename = os.path.join(self.base_path, row.filename)
        label = row.target
        wav, sr = torchaudio.load(filename, normalize=True)
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)(wav)
        mel_spec = self.mel_spectrogram(wav)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return mel_spec_db, label

# dataset = ESC50Dataset('../ESC-50/audio', meta_data)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# mel, label = dataset[21]
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 4))
# plt.imshow(mel.squeeze().numpy(), cmap='hot', origin='lower', aspect='auto')
# plt.title(f'Mel-Spectrogram of Example with Label: {label}')
# plt.xlabel('Time')
# plt.ylabel('Mel Filter Bank')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.savefig('log_mel.png')
