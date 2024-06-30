#(2) Adding background noise & speech

import numpy as np
import os
import random
import soundfile as sf
import librosa
import tqdm
class NoiseDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        root_dir=self.folder_path
        #root_dir = '/home/marinjezidzic/Downloads/LibriSpeech/test-clean'
        #self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
        self.files = [os.path.join(root, file) for root, dirs, files in os.walk(root_dir) for file in files if file.endswith(('.wav', '.flac'))]
        if not self.files:
            raise ValueError("No WAV files found in the specified folder.")

    def random(self, dtype=np.float32, duration_sec=2):
        # Select a random file
        random_file = random.choice(self.files)
        # Load a random segment from the selected file
        data, samplerate = sf.read(random_file)
        max_start = len(data) - samplerate * duration_sec
        start = random.randint(0, max_start)
        return data[start:start + samplerate * duration_sec].astype(dtype)
    def __len__(self):
        return len(self.files)

class Engine:
    @staticmethod
    def frame_length():
        return 1024  # Placeholder for frame length, adjust as needed.


def _pcm_energy(pcm):
    frame_length = Engine.frame_length()
    num_frames = pcm.size // frame_length
    pcm_frames = pcm[:num_frames * frame_length].reshape(num_frames, frame_length)
    frames_power = (pcm_frames ** 2).sum(axis=1)
    return frames_power.max()

def _speech_scale(speech, noise, snr_db):
    assert speech.shape[0] == noise.shape[0]
    speech_energy = _pcm_energy(speech)
    if speech_energy == 0:
        return 0
    return np.sqrt((_pcm_energy(noise) * (10 ** (snr_db / 10))) / speech_energy)

def _max_abs(x):
    return max(np.max(x), np.abs(np.min(x)))

def _mix_noise(speech_parts, noise_dataset, snr_db):
    speech_length = sum(len(x) for x in speech_parts)
    parts = []
    while sum(x.size for x in parts) < speech_length:
        x = noise_dataset.random()
        parts.append(x / _max_abs(x))

    res = np.concatenate(parts)[:speech_length]
    start_index = 0
    for speech_part in speech_parts:
        end_index = start_index + len(speech_part)
        res[start_index:end_index] += speech_part * _speech_scale(speech_part, res[start_index:end_index], snr_db)
        start_index = end_index

    return res


Data_1 = NoiseDataset(folder_path='/home/ai01/Desktop/marin_work_folder/FSDnoisy18k.audio_train/')
Data_2 = NoiseDataset(folder_path = '/mnt/nvme_ssd_2/train_10000')

folder_path = '/home/ai01/Desktop/marin_work_folder/extra_cf_samples'
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
out = '/home/ai01/Desktop/marin_work_folder/extra_cf_samples_aug/'
for i,file in tqdm.tqdm(enumerate(files)):
    y,sr = librosa.load(file,sr=None)
    snr_1 = random.randint(15, 55)
    snr_2 = random.randint(35, 65)
    try:
        mixed_signal_1 = _mix_noise([y], Data_1, snr_1)
        mixed_signal_2 = _mix_noise([y], Data_2, snr_2)
        sf.write(f'{out}_{i}{1}.wav', mixed_signal_1, 16000)
        sf.write(f'{out}_{i}{2}.wav', mixed_signal_2, 16000)
        print('saved')
    except:
        continue;