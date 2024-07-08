import torchaudio
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import random
import torchvision.transforms as transforms
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group,destroy_process_group
import argparse
import sys


# -----------------------------------------------------
# Setting up ddp
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"]= ""
    os.environ["MASTER_PORT"]= ""
    init_process_group(backend="nccl",rank=rank,world_size=world_size)






# ------------------------------------------------------
# Preprocessing steps
def set_random_rows_to_zero(image_tensor, max_consecutive_rows):
    height, width = image_tensor.shape[0], image_tensor.shape[1]
    # Handle case where the height is less than the max consecutive rows
    if height <= max_consecutive_rows:
        num_rows = height  # Use the entire height if it's less than or equal to max rows
    else:
        num_rows = random.randint(0, max_consecutive_rows)
    
    if num_rows > 0:  # Only modify the tensor if num_rows is greater than zero
        start_row = random.randint(0, height - num_rows)
        image_tensor[start_row:start_row + num_rows, :] = 0

    return image_tensor

def set_random_columns_to_zero(image_tensor, max_consecutive_columns):
    height, width = image_tensor.shape[0], image_tensor.shape[1]
    # Handle case where the width is less than the max consecutive columns
    if width <= max_consecutive_columns:
        num_columns = width  # Use the entire width if it's less than or equal to max columns
    else:
        num_columns = random.randint(0, max_consecutive_columns)
    
    if num_columns > 0:  # Only modify the tensor if num_columns is greater than zero
        start_column = random.randint(0, width - num_columns)
        image_tensor[:, start_column:start_column + num_columns] = 0

    return image_tensor

def apply_spectral_augment(image, num):
    if num == 1:
        aug_image = set_random_rows_to_zero(image, 20)  # 20 rows in the frequency domain
        return aug_image
    elif num == 2:
        aug_image = set_random_columns_to_zero(image, 30)  # 30 columns in the time domain
        return aug_image
    else:
        aug_image1 = set_random_rows_to_zero(image, 20)
        aug_image = set_random_columns_to_zero(aug_image1, 30)
        return aug_image




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
    waveform,sr = torchaudio.load(audio_path)
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    #mel_spec_normalized = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-8)
    transform = transforms.Compose([
        transforms.Normalize(mean=[mel_spec_db.mean()], std=[mel_spec_db.std()+1e-8]),
        transforms.Resize((256, 200))
    ])

    mel_spec_transformed = transform(mel_spec_db)
    
    return mel_spec_transformed


def compute_mel_spectrogram_aug(audio_path,mel_spectrogram,amplitude_to_db,   n_mels=256, sr=16000):
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
    waveform = waveform
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    #mel_spec_normalized = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-8)
    transform = transforms.Compose([
        transforms.Normalize(mean=[mel_spec_db.mean()], std=[mel_spec_db.std()+1e-8]), # Clamp standard deviation
        transforms.Resize((256, 200))
    ])

    mel_spec_transformed = transform(mel_spec_db)
    
    num = random.randint(1,3)
    aug_mel = apply_spectral_augment(mel_spec_transformed,num)
    return aug_mel

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
    #print(correct.size())
    return correct.sum()   # Return number of correct classifications





def train_one_epoch(model,loss_function,scheduler,optimizer,train_loader,hparams,device):
    train_loss=0
    model.train()
    for i, (feats, labels) in enumerate(train_loader):
        t0 = time.time()
        # Your training process here
        optimizer.zero_grad()
        features=feats.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs= model(features)
            labels = labels.float()
            labels=labels.to(device)
            loss = loss_function(torch.squeeze(outputs), labels )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['clip_norm'])
        optimizer.step()
        train_loss+=loss.cpu().item()
        torch.cuda.synchronize()
        t1 = time.time()

        if device==0 and i%100==0:
            print(f"Batch size: {hparams['batch_size']} | step/total_steps: {i//hparams['num_devices']} / {len(train_loader)//hparams['num_devices']} | loss: {loss.cpu().item():.6f} | dt: {(t1-t0)*1000:.2f}ms | norm: {norm:.4f} | lr: {optimizer.param_groups[0]['lr']}")
        torch.cuda.empty_cache()
    scheduler.step()
    return train_loss/len(train_loader)
    
def valid_one_epoch(model,loss_function,valid_loader,hparams,device):
    valid_loss=0
    total_num_correct = 0
    model.eval()
    for i, (feats, labels) in tqdm.tqdm(enumerate(valid_loader)):
        # Your training process here
        #print(f"{i} and label is {labels} ")
        features=feats.to(device)
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            outputs= model(features)
        labels = labels.float()
        labels=labels.to(device)
        loss = loss_function(torch.squeeze(outputs), labels)
        batch_num_correct = compute_accuracy(torch.squeeze(torch.sigmoid(outputs)), labels)
        total_num_correct += batch_num_correct
        valid_loss+=loss.cpu().item()
    average_accuracy = total_num_correct / ( (i+1)*hparams['batch_size'])
    #print(f'Validation Accuracy: {average_accuracy:.2f}')    
    return valid_loss/len(valid_loader) ,average_accuracy

def trainer(rank, world_size,hparams):
    dist.barrier() # sync all devices here
    csv_file = hparams['train_csv_file']
    csv_file_stage2 = hparams['train_csv_file_stage2']
    testing_csv_file=hparams['test_csv_file']
    device = rank
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    file = pd.read_csv(csv_file)
    test_file=pd.read_csv(testing_csv_file)
    if csv_file_stage2:
        trainstage2_file = pd.read_csv(csv_file_stage2)
        train_dataset_stage2 = AudioDataset(csv_file=trainstage2_file[trainstage2_file['split']=='train'],device=device,transform=compute_mel_spectrogram)
        train_loader_stage2 = DataLoader(train_dataset_stage2, batch_size=hparams['batch_size'], shuffle=False,num_workers=4,sampler=DistributedSampler(train_dataset_stage2),drop_last=True)

    train_dataset = AudioDataset(csv_file=file[file['split']=='train'],device=device,transform=compute_mel_spectrogram_aug)
    valid_dataset = AudioDataset(csv_file=file[file['split']=='valid'].reset_index(drop=True),device=device,transform=compute_mel_spectrogram)
    test_dataset = AudioDataset(csv_file=test_file,device=device,transform=compute_mel_spectrogram)
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False,num_workers=4,sampler=DistributedSampler(train_dataset),drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=hparams['batch_size'], shuffle=False,drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False,drop_last=False)

    torch.set_float32_matmul_precision('high')

    if hparams['use_version_1']:
        from models.se_res2net_1 import se_res2net50_1
        model_slow = se_res2net50_1(num_classes=1, in_channels=1)
    else:
        from models.se_res2net_2 import se_res2net50_2
        model_slow = se_res2net50_2(num_classes=1, in_channels=1)

    if hparams['pretrained_model_path']:
        state_dict = torch.load(hparams['pretrained_model_path']) # this should be loaded to CUDA:0 by default and but it does not matter since we transfer it to 'rank' device later on.
        model_slow.load_state_dict(state_dict)
    
    model_slow =model_slow.to(rank)
    

    
    device=rank

    if device==0:
        print(f"Number of parameters of model is:{sum(p.numel() for p in model_slow.parameters())}")

    ddp_mp_model = torch.compile(model_slow)
    ddp_mp_model = DDP(ddp_mp_model, device_ids=[rank])
    
    # Optimizer config
    optimizer = optim.Adam(ddp_mp_model.parameters(), lr=hparams['lr'])
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # Defining loss function
    loss_function = nn.BCEWithLogitsLoss()


    for epoch in range(hparams['num_stage_1_epochs']):

        if device==0:
            print("Initiating first stage of training")
        
        train_loss=train_one_epoch(ddp_mp_model,loss_function,scheduler,optimizer,train_loader,hparams,device)
        valid_loss,valid_acc = valid_one_epoch(ddp_mp_model,loss_function,valid_loader,hparams,device)
        _,test_acc = valid_one_epoch(ddp_mp_model,loss_function,test_loader,hparams,device)
        if device==0:
            print(f'Validation Accuracy: {valid_acc:.2f}')
            print(f'Device: {device} Test Accuracy: {test_acc:.2f}')
            ckp = ddp_mp_model.module.state_dict()
            torch.save(ckp, f'{hparams["save_path"]}_{epoch+1}.pt')
            print(f'Model saved for checkpoint:{epoch+1} at location {hparams["save_path"]}_{epoch+1}.pt')
            print(f'Train loss at epoch:{epoch+1} is: {train_loss}')
            print(f'Valid loss at epoch:{epoch+1} is: {valid_loss}')
            print('---------------------------------/-------------------------------------')
    if csv_file_stage2:
        for epoch in range(hparams['num_stage_2_epochs']):
            if device==0:
                print("Initiating second stage of training")
            train_loss_stage2=train_one_epoch(ddp_mp_model,loss_function,scheduler,optimizer,train_loader_stage2,hparams,device)
            valid_loss,valid_acc = valid_one_epoch(ddp_mp_model,loss_function,valid_loader,hparams,device)
            _,test_acc = valid_one_epoch(ddp_mp_model,loss_function,test_loader,hparams,device)
            if device==0: # We perform validation and testing only on one gpu so as to have unambiguous results.
                print(f'Validation Accuracy: {valid_acc:.2f}')
                print(f'Test Accuracy: {test_acc:.2f}')
                ckp = ddp_mp_model.module.state_dict()
                torch.save(ckp, f'{hparams["save_path"]}_{hparams["num_stage_1_epochs"]+epoch}.pt')
                print(f'Model saved for checkpoint:{hparams["num_stage_1_epochs"]+epoch} at location {hparams["save_path"]}_{hparams["num_stage_1_epochs"]+epoch}.pt')
                print(f'Train loss at epoch:{hparams["num_stage_1_epochs"]+epoch} of second stage is: {train_loss_stage2}')
                print(f'Valid loss at epoch:{hparams["num_stage_1_epochs"]+epoch} of second stage is: {valid_loss}')
                print('---------------------------------/-------------------------------------')



def main(rank: int, world_size: int, hparams: dict):
    ddp_setup(rank,world_size)
    trainer(rank,world_size,hparams)
    destroy_process_group()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_file',type=str, help='Metadata for training set first stage')
    parser.add_argument('--train_csv_file_stage2',type=str, help='Metadata for training set second stage',default=None)
    parser.add_argument('--test_csv_file',type=str, help='Metadata for testing set')
    parser.add_argument('--ngpu',type=int, help='Number of graphical units to use',default=1)
    parser.add_argument('--batch_size',type=int, help='Batch size during training phase',default=256)
    parser.add_argument('--num_stage_1_epochs',type=int, help='Batch size during training phase')
    parser.add_argument('--num_stage_2_epochs',type=int, help='Batch size during training phase',default=None)
    parser.add_argument('--pretrained_model_path',type=str, help='Path for pretrained model',default=None)
    parser.add_argument('--clip_norm',type=float, help='Norm to which to clip gradients',default=1.0)
    parser.add_argument('--lr',type=float, help='Stock value for learning rate',default=3e-4)
    parser.add_argument('--save_path',type=str, help='Path where to save checkpoints',default='')
    parser.add_argument('--use_version_1',type=bool, help='Version 1 is bigger model, same as in paper',default=False)

    

    args = parser.parse_args()
    
    hparams={
    'train_csv_file':args.train_csv_file,
    'train_csv_file_stage2': args.train_csv_file_stage2,
    'test_csv_file': args.test_csv_file,
    'batch_size':args.batch_size,
    'num_stage_1_epochs':args.num_stage_1_epochs,
    'num_stage_2_epochs':args.num_stage_2_epochs,
    'num_devices': args.ngpu,
    'pretrained_model_path': args.pretrained_model_path,
    'clip_norm': args.clip_norm,
    'lr': args.lr,
    'save_path':args.save_path,
    'use_version_1':args.use_version_1,
    }
    
    mp.spawn(main,args=(args.ngpu,hparams),nprocs=args.ngpu)
