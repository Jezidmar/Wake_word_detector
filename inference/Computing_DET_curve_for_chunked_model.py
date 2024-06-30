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

def test_file(feats,model,threshold=0.5):
    start_frame=0
    end_frame=feats.shape[3]
    probs=[]
    #print(end_frame)
    if end_frame<121:
        #pad to 121
        feats = pad_feats(feats,121-end_frame)
        end_frame=121
    while start_frame+121<=end_frame:
        chunk=feats[:,:,:,start_frame:start_frame+121] # 1.2 seconds WINDOW size
        #print(chunk.shape)
        start_frame+=20 #Step size is 0.2 seconds ~ 20 frames
        probs.append(torch.sigmoid(model( chunk)))
    out_prob = torch.mean( torch.tensor(probs) )
    if out_prob>threshold:
        return 1
    else:
        return 0

import numpy as np

def pad_feats(mel_spec, pad_amount):
    """
    Pads the mel spectrogram to a maximum length with a given pad value.

    Args:
    mel_spec (np.array): The mel spectrogram array (time x features).
    max_length (int): The maximum length to pad the array.
    pad_value (float, optional): The value used for padding. Defaults to 0.

    Returns:
    np.array: Padded mel spectrogram.
    """
    # Calculate the amount of padding needed
    padding_pattern = (0, pad_amount)
    pad_value=0
    adjusted_tensor = torch.nn.functional.pad(mel_spec, padding_pattern, value=pad_value)


    return adjusted_tensor

def test_one_epoch_c(model, test_loader, hparams):
    model.eval()
    device = hparams['device']
    total_test_time_seconds = 0
    # Initialize detection error tradeoff (DET) metrics
    thresholds = np.arange(0.03, 1.01, 0.03)  # Including 1.00 for completeness
    # Keeping track of true positives, false positives, false negatives, true negatives
    tp = np.zeros_like(thresholds)
    fp = np.zeros_like(thresholds)
    fn = np.zeros_like(thresholds)
    tn = np.zeros_like(thresholds)

    # Iterate over all batches in the test loader
    for paths, labels in tqdm.tqdm(test_loader):
        #feats = feats.unsqueeze(1).to(device)  # Ensure input dimensions and device are correct
        feats = extract_mel_spectrograms(paths,device)
        # Compute model probabilities for the entire batch
        end_frame = feats.shape[3]
        if end_frame < 121:
            feats = pad_feats(feats, 121 - end_frame)
            end_frame = 121


        batch_duration = feats.shape[3] / hparams['sample_rate']  # Assuming the third dimension is time
        total_test_time_seconds += batch_duration
        
        start_frame = 0
        probs = []
        while start_frame + 121 <= end_frame:
            chunk = feats[:, :, :, start_frame:start_frame + 121]  # 1.2 seconds window size
            start_frame += 20  # Step size is 0.2 seconds ~ 20 frames
            probs.append(torch.sigmoid(model(chunk)))
        
        probs = torch.mean(torch.stack(probs), dim=0)  # Average probabilities over chunks
        
        # Compare predictions against each threshold
        for idx, thresh in enumerate(thresholds):
            pred = (probs > thresh).int()
            labels = labels.int()
            tp[idx] += ((pred == 1) & (labels == 1)).sum().item()
            fp[idx] += ((pred == 1) & (labels == 0)).sum().item()
            fn[idx] += ((pred == 0) & (labels == 1)).sum().item()
            tn[idx] += ((pred == 0) & (labels == 0)).sum().item()


    # Calculate metrics needed for DET curve
    far = fp / (fp + tn)  # False Alarm Rate
    frr = fn / (fn + tp)  # False Rejection Rate (Miss Rate)
    hours = total_test_time_seconds / 3600
    far_per_hour = fp / hours
    # Optionally, return the results for further analysis or plotting
    return {'Thresholds': thresholds, 'FAR': far, 'FRR': frr, 'FAR_per_Hour': far_per_hour, 'Total Testing Time (hours)': hours}



def run_c(hparams):
    testing_csv_file=hparams['test_csv_file']
    test_file=pd.read_csv(testing_csv_file)
    test_dataset = AudioDataset(csv_file=test_file,transform=get_mel_spectrogram)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = load_model(hparams['checkpoint_path'])
    metrics=test_one_epoch_c(model, test_loader, hparams)
    return metrics
hparams={
    'test_csv_file': '/home/ai01/Desktop/marin_work_folder/train_code/metadata_test_clean.csv',
    'device': 'cpu',
    'checkpoint_path' : 'checkpoints/checkpoint_5_trial4.pth',   #<-- trial on dataset v1
    'checkpoint_path' : 'checkpoints/checkpoint_resnet_small_18_trial1_chunked.pth',
    'sample_rate':16000
}
metrics=run_c(hparams)


def export_metrics_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Metrics exported successfully to {filename}")

metrics=run_c(hparams)
export_metrics_to_csv(metrics, 'test_metrics_c_res2net_clean_torch.csv')