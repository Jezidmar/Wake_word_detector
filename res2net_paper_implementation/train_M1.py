import argparse
import os
import time

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchaudio
import tqdm
from data.tools.extract_feats import (
    compute_mel_spectrogram,
    compute_mel_spectrogram_aug,
)
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# -----------------------------------------------------
# Setting up ddp
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = ""
    os.environ["MASTER_PORT"] = ""
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


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
        with torch.no_grad():
            probs.append(torch.sigmoid(model(chunk)))
    out_prob = torch.mean(torch.tensor(probs))
    if out_prob > threshold:
        return 1
    else:
        return 0


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


def compute_accuracy(outputs, labels, threshold=0.5):
    """
    Compute accuracy given model outputs and ground truth labels.

    Args:
    outputs: Model outputs (raw logits or probabilities).
    labels: Ground truth labels.
    threshold: Threshold value for binary classification.

    Returns:
    accuracy: Accuracy value.
    """
    preds = (outputs > threshold).type(torch.float32)  # Convert boolean to float
    correct = (preds == labels).type(torch.float32)  # Convert boolean to float
    # print(correct.size())
    return correct.sum()  # Return number of correct classifications


def train_one_epoch(
    model, loss_function, scheduler, optimizer, train_loader, args, device
):
    train_loss = 0
    model.train()
    for i, (feats, labels) in enumerate(train_loader):
        t0 = time.time()
        # Your training process here
        optimizer.zero_grad()
        features = feats.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(features)
            labels = labels.float()
            labels = labels.to(device)
            loss = loss_function(torch.squeeze(outputs), labels)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        train_loss += loss.cpu().item()
        torch.cuda.synchronize()
        t1 = time.time()

        if device == 0 and i % 100 == 0:
            print(
                f"Batch size: {args.batch_size} | step/total_steps: {i//args.num_devices} / {len(train_loader)//args.num_devices} | loss: {loss.cpu().item():.6f} | dt: {(t1-t0)*1000:.2f}ms | norm: {norm:.4f} | lr: {optimizer.param_groups[0]['lr']}"
            )
        torch.cuda.empty_cache()
    scheduler.step()
    return train_loss / len(train_loader)


def valid_one_epoch(model, loss_function, valid_loader, args, device):
    valid_loss = 0
    total_num_correct = 0
    model.eval()
    for i, (feats, labels) in tqdm.tqdm(enumerate(valid_loader)):
        # Your training process here
        # print(f"{i} and label is {labels} ")
        features = feats.to(device)
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            outputs = model(features)
        labels = labels.float()
        labels = labels.to(device)
        loss = loss_function(torch.squeeze(outputs), labels)
        batch_num_correct = compute_accuracy(
            torch.squeeze(torch.sigmoid(outputs)), labels
        )
        total_num_correct += batch_num_correct
        valid_loss += loss.cpu().item()
    average_accuracy = total_num_correct / ((i + 1) * args.batch_size)
    # print(f'Validation Accuracy: {average_accuracy:.2f}')
    return valid_loss / len(valid_loader), average_accuracy


def test_one_epoch(model, test_loader, args):
    device = args.device
    model.eval()
    total_num_correct = 0
    for i, (feats, labels) in enumerate(test_loader):
        features = feats.unsqueeze(1).to(device)
        pred = test_file(features, model)
        # print(f"If {pred} and {torch.squeeze(labels)}, then {pred==torch.squeeze(labels)}")
        if pred == torch.squeeze(labels):
            total_num_correct += 1
    return total_num_correct / i


def trainer(rank, world_size, args):
    dist.barrier()  # sync all devices here
    csv_file = args.train_csv_file
    csv_file_stage2 = args.train_csv_file_stage2
    testing_csv_file = args.test_csv_file
    device = rank
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    file = pd.read_csv(csv_file)
    test_file = pd.read_csv(testing_csv_file)
    if csv_file_stage2:
        trainstage2_file = pd.read_csv(csv_file_stage2)
        train_dataset_stage2 = AudioDataset(
            csv_file=trainstage2_file[trainstage2_file["split"] == "train"],
            device=device,
            transform=compute_mel_spectrogram,
        )
        train_loader_stage2 = DataLoader(
            train_dataset_stage2,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            sampler=DistributedSampler(train_dataset_stage2),
            drop_last=True,
        )

    train_dataset = AudioDataset(
        csv_file=file[file["split"] == "train"],
        device=device,
        transform=compute_mel_spectrogram_aug,
    )
    valid_dataset = AudioDataset(
        csv_file=file[file["split"] == "valid"].reset_index(drop=True),
        device=device,
        transform=compute_mel_spectrogram,
    )
    test_dataset = AudioDataset(
        csv_file=test_file, device=device, transform=compute_mel_spectrogram
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(train_dataset),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    torch.set_float32_matmul_precision("high")

    if args.use_version_1:
        from models.se_res2net_1 import se_res2net50_1

        model_slow = se_res2net50_1(num_classes=1, in_channels=1)
    else:
        from models.se_res2net_2 import se_res2net50_2

        model_slow = se_res2net50_2(num_classes=1, in_channels=1)

    if args.pretrained_model_path:
        state_dict = torch.load(
            args.pretrained_model_path
        )  # this should be loaded to CUDA:0 by default and but it does not matter since we transfer it to 'rank' device later on.
        model_slow.load_state_dict(state_dict)

    model_slow = model_slow.to(rank)

    device = rank

    if device == 0:
        print(
            f"Number of parameters of model is:{sum(p.numel() for p in model_slow.parameters())}"
        )

    ddp_mp_model = torch.compile(model_slow)
    ddp_mp_model = DDP(ddp_mp_model, device_ids=[rank])

    # Optimizer config
    optimizer = optim.Adam(ddp_mp_model.parameters(), lr=args.lr)
    lambda1 = lambda epoch: 0.65**epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # Defining loss function
    loss_function = nn.BCEWithLogitsLoss()

    for epoch in range(args.num_stage_1_epochs):
        if device == 0:
            print("Initiating first stage of training")

        train_loss = train_one_epoch(
            ddp_mp_model,
            loss_function,
            scheduler,
            optimizer,
            train_loader,
            args,
            device,
        )
        valid_loss, valid_acc = valid_one_epoch(
            ddp_mp_model, loss_function, valid_loader, args, device
        )
        test_acc = test_one_epoch(
            ddp_mp_model, loss_function, test_loader, args, device
        )
        if device == 0:
            print(f"Validation Accuracy: {valid_acc:.2f}")
            print(f"Device: {device} Test Accuracy: {test_acc:.2f}")
            ckp = ddp_mp_model.module.state_dict()
            torch.save(ckp, f"{args.save_path}_{epoch+1}.pt")
            print(
                f"Model saved for checkpoint:{epoch+1} at location {args.save_path}_{epoch+1}.pt"
            )
            print(f"Train loss at epoch:{epoch+1} is: {train_loss}")
            print(f"Valid loss at epoch:{epoch+1} is: {valid_loss}")
            print(
                "---------------------------------/-------------------------------------"
            )
    if csv_file_stage2:
        for epoch in range(args.num_stage_2_epochs):
            if device == 0:
                print("Initiating second stage of training")
            train_loss_stage2 = train_one_epoch(
                ddp_mp_model,
                loss_function,
                scheduler,
                optimizer,
                train_loader_stage2,
                args,
                device,
            )
            valid_loss, valid_acc = valid_one_epoch(
                ddp_mp_model, loss_function, valid_loader, args, device
            )
            test_acc = test_one_epoch(ddp_mp_model, test_loader, args, device)
            if (
                device == 0
            ):  # We perform validation and testing only on one gpu so as to have unambiguous results.
                print(f"Validation Accuracy: {valid_acc:.2f}")
                print(f"Test Accuracy: {test_acc:.2f}")
                ckp = ddp_mp_model.module.state_dict()
                torch.save(
                    ckp,
                    f"{args.save_path}_{args.num_stage_1_epochs+epoch}.pt",
                )
                print(
                    f"Model saved for checkpoint:{args.num_stage_1_epochs+epoch} at location {args.save_path}_{args.num_stage_1_epochs+epoch}.pt"
                )
                print(
                    f"Train loss at epoch:{args.num_stage_1_epochs+epoch} of second stage is: {train_loss_stage2}"
                )
                print(
                    f"Valid loss at epoch:{args.num_stage_1_epochs+epoch} of second stage is: {valid_loss}"
                )
                print(
                    "---------------------------------/-------------------------------------"
                )


def main(rank: int, world_size: int, args: dict):
    ddp_setup(rank, world_size)
    trainer(rank, world_size, args)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv_file", type=str, help="Metadata for training set first stage"
    )
    parser.add_argument(
        "--train_csv_file_stage2",
        type=str,
        help="Metadata for training set second stage",
        default=None,
    )
    parser.add_argument("--test_csv_file", type=str, help="Metadata for testing set")
    parser.add_argument(
        "--ngpu", type=int, help="Number of graphical units to use", default=1
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size during training phase", default=256
    )
    parser.add_argument(
        "--num_stage_1_epochs", type=int, help="Batch size during training phase"
    )
    parser.add_argument(
        "--num_stage_2_epochs",
        type=int,
        help="Batch size during training phase",
        default=None,
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        help="Path for pretrained model",
        default=None,
    )
    parser.add_argument(
        "--clip_norm", type=float, help="Norm to which to clip gradients", default=1.0
    )
    parser.add_argument(
        "--lr", type=float, help="Stock value for learning rate", default=3e-4
    )
    parser.add_argument(
        "--save_path", type=str, help="Path where to save checkpoints", default=""
    )
    parser.add_argument(
        "--use_version_1",
        type=bool,
        help="Version 1 is bigger model, same as in paper",
        default=False,
    )

    args = parser.parse_args()

    mp.spawn(main, args=(args.ngpu, args), nprocs=args.ngpu)
