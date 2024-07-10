import argparse
import struct
import wave

import numpy as np
import pandas as pd
import pvporcupine
import tqdm
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        audio_path = self.data_frame["path"][idx]
        label = self.data_frame["label"][idx]

        return audio_path, label


def export_metrics_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Metrics exported successfully to {filename}")


def process_audio(porcupine, wf):
    # Buffer to hold audio data
    audio_buffer = bytes()
    total_frames_processed = 0

    while True:
        # Read data from the file
        data = wf.readframes(porcupine.frame_length)
        if len(data) == 0:
            break  # End of file

        # Append data to buffer
        audio_buffer += data

        # Check if buffer has enough data to process
        if (
            len(audio_buffer) >= porcupine.frame_length * 2
        ):  # 2 bytes per frame (16-bit audio)
            # Process the audio in frame_length chunks
            pcm = struct.unpack_from("h" * porcupine.frame_length, audio_buffer, 0)
            result = porcupine.process(pcm)
            if result >= 0:
                return result

            # Move the buffer forward
            audio_buffer = audio_buffer[porcupine.frame_length * 2 :]
            total_frames_processed += porcupine.frame_length

    # Process any remaining audio data by padding if necessary
    if len(audio_buffer) > 0:
        padded_pcm = struct.unpack_from(
            "h" * porcupine.frame_length,
            audio_buffer + b"\x00" * (porcupine.frame_length * 2 - len(audio_buffer)),
            0,
        )
        result = porcupine.process(padded_pcm)
        if result >= 0:
            return result
    return -1


def process(file_path, thresh, args):
    porcupine = pvporcupine.create(
        access_key=args.access_key,
        keyword_paths=[args.keyword_path],
        sensitivities=[thresh],
    )

    wf = wave.open(file_path, "rb")
    val = process_audio(porcupine, wf)
    return val


def run(args):
    test_file = pd.read_csv(args.test_csv_file)
    test_dataset = AudioDataset(csv_file=test_file)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    thresholds = np.arange(0.03, 1.01, 0.03)
    tp = np.zeros_like(thresholds)
    fp = np.zeros_like(thresholds)
    fn = np.zeros_like(thresholds)
    tn = np.zeros_like(thresholds)
    for path, labels in tqdm.tqdm(test_loader):
        # print(path)
        for idx, thresh in enumerate(thresholds):
            pred = process(path[0], thresh) + 1
            labels = labels.int()
            tp[idx] += ((pred == 1) & (labels == 1)).sum().item()
            fp[idx] += ((pred == 1) & (labels == 0)).sum().item()
            fn[idx] += ((pred == 0) & (labels == 1)).sum().item()
            tn[idx] += ((pred == 0) & (labels == 0)).sum().item()

    # Calculate metrics needed for DET curve
    far = fp / (fp + tn)  # False Alarm Rate
    frr = fn / (fn + tp)  # False Rejection Rate (Miss Rate)

    return {"Thresholds": thresholds, "FAR": far, "FRR": frr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_csv_file", type=str, required=True, help="Metadata for testing set"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path where to save the obtained results in .csv file",
    )
    parser.add_argument(
        "--access_key", type=str, required=True, help="Access key for porcupine"
    )
    parser.add_argument(
        "--keyword_path", type=str, required=True, help="Path to keyword model"
    )

    args = parser.parse_args()

    metrics = run(args)
    export_metrics_to_csv(metrics, f"{args.save_path}results_porcupine.csv")
