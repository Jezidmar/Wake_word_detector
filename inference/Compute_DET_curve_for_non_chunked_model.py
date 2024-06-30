import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
import torchaudio
import librosa
import cv2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import numpy as np
import librosa
import logging
import soundfile
import torch.nn.functional as F
import csv
import tqdm
from model_res2net50_v1b import res2net50_v1b as resnet50


def compute_mel_spectrogram(audio_path,mel_spectrogram,amplitude_to_db,   n_mels=256, sr=16000):
    """
    Compute a Mel spectrogram for an audio file.

    Args:
    audio_path (str): Path to the audio file.
    n_fft (int): Number of FFT components.
    hop_length (int): Number of samples between successive frames.
    n_mels (int): Number of Mel bands to generate.
    sr (int): Sampling rate of the audio file.

    Returns:
    S_DB (np.array): Log-scaled Mel spectrogram.
    """
    # Load the audio file
    waveform,_ = torchaudio.load(audio_path)
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    transform = transforms.Compose([
        transforms.Normalize(mean=[mel_spec_db.mean()], std=[mel_spec_db.std()+1e-8]),
        transforms.Resize((256, 200))
    ])

    mel_spec_transformed = transform(mel_spec_db)
    
    return mel_spec_transformed


class AudioDataset(Dataset):
    def __init__(self, csv_file,device='cuda', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = csv_file
        self.transform = transform
        self.device=device
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,
            n_mels=256
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        audio_path = self.data_frame['path'][idx]
        label = self.data_frame['label'][idx]
        
        if self.transform:
            LogMel = self.transform(audio_path,self.mel_spectrogram,self.amplitude_to_db)

        return LogMel, label


def test_one_epoch_nc(model, test_loader, hparams):
    model.eval()
    device = hparams['device']

    thresholds = np.arange(0.03, 1.01, 0.03)
    tp = np.zeros_like(thresholds)
    fp = np.zeros_like(thresholds)
    fn = np.zeros_like(thresholds)
    tn = np.zeros_like(thresholds)

    total_test_time_seconds = 0  # Initialize total testing time accumulator

    for feats, labels in tqdm.tqdm(test_loader):
        feats = feats.unsqueeze(1).to(device)  # Ensure input dimensions and device are correct
        
        # Check if padding is needed


        # Compute model probabilities for the entire batch
        probs = torch.sigmoid(model(feats)).squeeze()  # Assume the model outputs raw logits
        
        # Calculate the total duration if necessary
        batch_duration = feats.shape[3] / hparams['sample_rate']  # Assuming the third dimension is time
        total_test_time_seconds += batch_duration
        
        # Threshold comparison
        labels = labels.to(device).int()  # Ensure labels are on the correct device and format
        for idx, thresh in enumerate(thresholds):
            pred = (probs > thresh).int()  # Apply threshold
            tp[idx] += ((pred == 1) & (labels == 1)).sum().item()
            fp[idx] += ((pred == 1) & (labels == 0)).sum().item()
            fn[idx] += ((pred == 0) & (labels == 1)).sum().item()
            tn[idx] += ((pred == 0) & (labels == 0)).sum().item()

    # Calculate metrics for DET curve
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    hours = total_test_time_seconds / 3600
    far_per_hour = fp / hours

    return {'Thresholds': thresholds, 'FAR': far, 'FRR': frr, 'FAR_per_Hour': far_per_hour, 'Total Testing Time (hours)': hours}

def run_nc(hparams):
    testing_csv_file=hparams['test_csv_file']
    test_file=pd.read_csv(testing_csv_file)
    test_dataset = AudioDataset(csv_file=test_file,transform=compute_mel_spectrogram)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # Load the checkpoint
    checkpoint = torch.load(hparams['checkpoint_path'], map_location=hparams['device'])
    model = resnet50(num_classes=1)
    checkpoint_path = hparams['checkpoint_path']
    loaded = torch.load(checkpoint_path)
    state_dict=loaded.module.state_dict()
    model.load_state_dict(state_dict)
    metrics=test_one_epoch_nc(model, test_loader, hparams)
    return metrics
hparams={
    'test_csv_file': '/home/ai01/Desktop/marin_work_folder/train_code/metadata_test_other.csv',
    'device': 'cpu',
    'sample_rate':16000,
    'checkpoint_path' : 'checkpoints/res2net50_v1b_trial1_epoch_6.pth',
}
metrics=run_nc(hparams)

def export_metrics_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Metrics exported successfully to {filename}")

#metrics = test_one_epoch(model, test_loader, hparams)
export_metrics_to_csv(metrics, 'test_metrics_nc_other_res2net_trial1_non_avg.csv')