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


def test_file(feats, model, threshold=0.5):
    start_frame = 0
    end_frame = feats.shape[3]
    probs = []
    # print(end_frame)
    if end_frame < 121:
        # pad to 121
        feats = pad_feats(feats, 121 - end_frame)
        end_frame = 121
    while start_frame + 121 <= end_frame:
        chunk = feats[
            :, :, :, start_frame : start_frame + 121
        ]  # 1.2 seconds WINDOW size
        # print(chunk.shape)
        start_frame += 20  # Step size is 0.2 seconds ~ 20 frames
        probs.append(torch.sigmoid(model(chunk)))
    out_prob = torch.mean(torch.tensor(probs))
    if out_prob > threshold:
        return 1
    else:
        return 0


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
    pad_value = 0
    adjusted_tensor = torch.nn.functional.pad(
        mel_spec, padding_pattern, value=pad_value
    )

    return adjusted_tensor


def test_one_epoch_M0(model, test_loader, args):
    model.eval()
    device = args.device
    total_test_time_seconds = 0
    # Initialize detection error tradeoff (DET) metrics
    thresholds = np.arange(0.03, 1.01, 0.03)  # Including 1.00 for completeness
    # Keeping track of true positives, false positives, false negatives, true negatives
    tp = np.zeros_like(thresholds)
    fp = np.zeros_like(thresholds)
    fn = np.zeros_like(thresholds)
    tn = np.zeros_like(thresholds)

    # Iterate over all batches in the test loader
    for feats, labels in tqdm.tqdm(test_loader):
        # feats = feats.unsqueeze(1).to(device)  # Ensure input dimensions and device are correct
        feats = feats.unsqueeze(1).to(device)
        # Compute model probabilities for the entire batch
        end_frame = feats.shape[3]
        if end_frame < 121:
            feats = pad_feats(feats, 121 - end_frame)
            end_frame = 121

        batch_duration = (
            feats.shape[3] / 16000
        )  # Assuming the third dimension is time, hardcoded sampling rate to 16khz
        total_test_time_seconds += batch_duration

        start_frame = 0
        probs = []
        while start_frame + 121 <= end_frame:
            chunk = feats[
                :, :, :, start_frame : start_frame + 121
            ]  # 1.2 seconds window size
            start_frame += 20  # Step size is 0.2 seconds ~ 20 frames
            with torch.no_grad:
                probs.append(torch.sigmoid(model(chunk)))

        probs = torch.mean(
            torch.stack(probs), dim=0
        )  # Average probabilities over chunks

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
    return {
        "Thresholds": thresholds,
        "FAR": far,
        "FRR": frr,
        "FAR_per_Hour": far_per_hour,
        "Total Testing Time (hours)": hours,
    }


def run_M0(args):
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
    metrics = test_one_epoch_M0(model, test_loader, args)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_csv_file", type=str, required=True, help="Metadata for testing set"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint for M0 model",
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

    metrics = run_M0(args)
    export_metrics_to_csv(metrics, f"{args.save_path}results_M0.csv")
