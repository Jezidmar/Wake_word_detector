import torch.nn as nn
import torch
import torchaudio.transforms as transforms


class MinMaxNormalizeTransform(nn.Module):
    def forward(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val+1e-8)
    

def initialize_mel_spectrogram_chunked_transform(n_fft=1024, hop_length=160, n_mels=128, sample_rate=16000):
    mel_spectrogram = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    to_db = transforms.AmplitudeToDB()
    normalize = MinMaxNormalizeTransform()
    return torch.nn.Sequential(mel_spectrogram, to_db, normalize)

def initialize_mel_spectrogram_non_chunked_transform(n_fft=1024, hop_length=160, n_mels=128, sr=16000):
    mel_spectrogram = transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        win_length=n_fft,
        pad_mode="constant",
        center=True,
        mel_scale="slaney",
        onesided=True,
        norm='slaney'
    
    )
    return mel_spectrogram
