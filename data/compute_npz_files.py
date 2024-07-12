import argparse
import os

import numpy as np
import torchaudio
import tqdm
from tools.extract_feats import compute_mel_spectrogram, compute_mel_spectrogram_aug


def create_individual_npz_files(
    base_directory, output_directory, mel_spectrogram, amplitude_to_db, n_mels
):
    folders = os.listdir(base_directory)
    categories = {}
    for folder in folders:
        categories[folder] = 1 if "positive" in folder else 0

    for category, label in categories.items():
        directory_path = os.path.join(base_directory, category)
        files = os.listdir(directory_path)
        for file in tqdm.tqdm(files):
            if file.endswith(".wav") or file.endswith(".mp3") or file.endswith(".flac"):
                file_path = os.path.join(directory_path, file)
                feats = compute_mel_spectrogram(
                    file_path, mel_spectrogram, amplitude_to_db, n_mels
                )
                feats_aug = compute_mel_spectrogram_aug(
                    file_path, mel_spectrogram, amplitude_to_db, n_mels
                )
                output_path = output_directory + +str(category) + "/" + file + ".npz"
                np.savez_compressed(
                    output_path, feats=feats, feats_aug=feats_aug, label=label
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_audio_files",
        type=str,
        help="Path to audio files for which we want to extract feats",
    )
    parser.add_argument(
        "--save_path", type=str, help="Path where to save the extracted feats"
    )
    args = parser.parse_args()

    # hardcoded argument:
    n_mels = 256

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=160, n_mels=n_mels
    )  # Default config as described in paper
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    create_individual_npz_files(
        args.path_to_audio_files,
        args.save_path,
        mel_spectrogram,
        amplitude_to_db,
        n_mels,
    )
