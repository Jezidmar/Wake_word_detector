# (2) Adding background noise & speech

import argparse
import os
import random

import librosa
import numpy as np
import soundfile as sf
import tqdm


class NoiseDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(self.folder_path)
            for file in files
            if file.endswith((".wav", ".flac"))
        ]
        if not self.files:
            raise ValueError("No WAV files found in the specified folder.")

    def random(self, dtype=np.float32, duration_sec=2):
        # Select a random file
        random_file = random.choice(self.files)
        # Load a random segment from the selected file
        data, samplerate = sf.read(random_file)
        max_start = len(data) - samplerate * duration_sec
        start = random.randint(0, max_start)
        return data[start : start + samplerate * duration_sec].astype(dtype)

    def __len__(self):
        return len(self.files)


class Engine:
    @staticmethod
    def frame_length():
        return 1024  # Placeholder for frame length, adjust as needed.


def _pcm_energy(pcm):
    frame_length = Engine.frame_length()
    num_frames = pcm.size // frame_length
    pcm_frames = pcm[: num_frames * frame_length].reshape(num_frames, frame_length)
    frames_power = (pcm_frames**2).sum(axis=1)
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
        res[start_index:end_index] += speech_part * _speech_scale(
            speech_part, res[start_index:end_index], snr_db
        )
        start_index = end_index

    return res


def perform_augmentation(
    Data_background_noise, Data_background_speech, files, output_path
):
    for i, file in tqdm.tqdm(enumerate(files)):
        y, sr = librosa.load(file, sr=None)
        snr_background_noise = random.randint(15, 55)
        snr_background_speech = random.randint(35, 65)
        mixed_signal_1 = _mix_noise([y], Data_background_noise, snr_background_noise)
        mixed_signal_2 = _mix_noise([y], Data_background_speech, snr_background_speech)
        sf.write(f"{output_path}_{i}_background_noise.wav", mixed_signal_1, 16000)
        sf.write(f"{output_path}_{i}_background_speech.wav", mixed_signal_2, 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_background_noise",
        type=str,
        help="Path to background noise audio files",
    )
    parser.add_argument(
        "--path_to_background_speech",
        type=str,
        help="Path to background speech files",
        default=None,
    )
    parser.add_argument(
        "--path_to_audio", type=str, help="Path to audio files for augmentation"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path where to save audio files after augmentation",
    )

    args = parser.parse_args()

    Data_1 = NoiseDataset(folder_path=args.path_to_background_noise)
    Data_2 = NoiseDataset(folder_path=args.path_to_background_speech)

    folder_path = args.path_to_audio
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".wav") or f.endswith(".flac") or f.endswith(".mp4")
    ]
    output_path = args.save_path
    perform_augmentation(Data_1, Data_2, files, output_path)
