import argparse

import numpy as np
import pandas as pd
import torch
import torchaudio
import tqdm
from data.tools.extract_feats import compute_mel_spectrogram
from torch.utils.data import DataLoader, Dataset


def export_metrics_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Metrics exported successfully to {filename}")


class AudioDataset(Dataset):
    def __init__(self, csv_file, device="cuda", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = csv_file
        self.transform = transform
        self.device = device
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=160, n_mels=256
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        audio_path = self.data_frame["path"][idx]
        label = self.data_frame["label"][idx]

        if self.transform:
            LogMel = self.transform(
                audio_path, self.mel_spectrogram, self.amplitude_to_db
            )

        return LogMel, label


def test_one_epoch_M1(model, test_loader, args):
    model.eval()
    device = args.device
    sample_rate = 16000
    thresholds = np.arange(0.03, 1.01, 0.03)
    tp = np.zeros_like(thresholds)
    fp = np.zeros_like(thresholds)
    fn = np.zeros_like(thresholds)
    tn = np.zeros_like(thresholds)

    total_test_time_seconds = 0  # Initialize total testing time accumulator

    for feats, labels in tqdm.tqdm(test_loader):
        feats = feats.unsqueeze(1).to(
            device
        )  # Ensure input dimensions and device are correct

        # Compute model probabilities for the entire batch
        with torch.no_grad():
            probs = torch.sigmoid(
                model(feats)
            ).squeeze()  # Assume the model outputs raw logits

        # Calculate the total duration if necessary
        batch_duration = (
            feats.shape[3] / sample_rate
        )  # Assuming the third dimension is time
        total_test_time_seconds += batch_duration

        # Threshold comparison
        labels = labels.to(
            device
        ).int()  # Ensure labels are on the correct device and format
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

    return {
        "Thresholds": thresholds,
        "FAR": far,
        "FRR": frr,
        "FAR_per_Hour": far_per_hour,
        "Total Testing Time (hours)": hours,
    }


def run_M1(args):
    test_file = pd.read_csv(args.test_csv_file)
    test_dataset = AudioDataset(csv_file=test_file, transform=compute_mel_spectrogram)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the checkpoint

    if args.use_version_1:
        from models.se_res2net_1 import se_res2net50_1

        model = se_res2net50_1(num_classes=1, in_channels=1)
    else:
        from models.se_res2net_2 import se_res2net50_2

        model = se_res2net50_2(num_classes=1, in_channels=1)

    checkpoint_path = args.checkpoint_path
    loaded = torch.load(checkpoint_path)
    state_dict = loaded.module.state_dict()
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    metrics = test_one_epoch_M1(model, test_loader, args)
    return metrics


# metrics = test_one_epoch(model, test_loader, hparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_csv_file", type=str, required=True, help="Metadata for testing set"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint for M1 model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path where to save the obtained results in .csv file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Which device will be used to perform inference",
    )
    parser.add_argument(
        "--use_version_1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use smaller(2) or larger(1) version, same as in paper",
    )

    args = parser.parse_args()

    metrics = run_M1(args)
    export_metrics_to_csv(metrics, f"{args.save_path}results_M1.csv")
