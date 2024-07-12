# (1) Concatenating
import argparse
import os
import random

import librosa
import numpy as np
import soundfile as sf
import tqdm


def concat_positive_samples(directory, output_dir, extra_percentage=0.2):
    # List all .wav files in the directory
    files = [
        f
        for f in os.listdir(directory)
        if f.endswith(".wav") or f.endswith(".flac") or f.endswith(".mp4")
    ]

    # Calculate the number of additional concatenated files to create (% of the original file count)
    extra_files_count = int(len(files) * extra_percentage)

    # Create the additional concatenated audio files
    for i in tqdm.tqdm(range(extra_files_count)):
        # Randomly decide whether to concatenate 2 or 3 files
        num_to_concatenate = random.choice([2, 3])
        files_to_concatenate = random.sample(files, num_to_concatenate)

        # Load and concatenate the audio files
        audio_data = []
        for file in files_to_concatenate:
            filepath = os.path.join(directory, file)
            data, _ = librosa.load(
                filepath, sr=16000
            )  # Sampling rate is hardcoded to 16K everywhere
            audio_data.append(data)

        # Concatenate along the time axis (axis=0)
        concatenated_data = np.concatenate(audio_data, axis=0)

        # Save the concatenated audio
        output_filename = f"positive_concatenated_{i}.wav"
        output_filepath = os.path.join(output_dir, output_filename)
        sf.write(output_filepath, concatenated_data, 16000)


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
    parser.add_argument(
        "--extra_percentage",
        type=str,
        help="What percentage of additional data you want to create relative to already existing amount",
    )
    args = parser.parse_args()

    concat_positive_samples(args.path_to_audio_files, args.save_path)
