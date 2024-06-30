import torchaudio
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from resnet_small import se_res2net50_v1b_non_chunked as resnet50
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


# DDP config
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"]= "172.16.170.6"
    os.environ["MASTER_PORT"]= "65521"
    init_process_group(backend="nccl",rank=rank,world_size=world_size)



def set_random_rows_to_zero(image_tensor, max_consecutive_rows):
    height, width = image_tensor.shape[0], image_tensor.shape[1]
    num_rows = random.randint(0, max_consecutive_rows)  # Randomly select the number of consecutive rows to set to zero
    start_row = random.randint(0, height - num_rows - 1)  # Randomly select the starting row index

    # Set the selected consecutive rows to zero
    image_tensor[ start_row:start_row + num_rows, :] = 0

    return image_tensor

def set_random_columns_to_zero(image_tensor, max_consecutive_columns):

    height, width = image_tensor.shape[0], image_tensor.shape[1]
    num_columns = random.randint(0, max_consecutive_columns)  # Randomly select the number of consecutive columns to set to zero
    start_column = random.randint(0, width - num_columns - 1)  # Randomly select the starting column index

    # Set the selected consecutive columns to zero
    image_tensor[ :, start_column:start_column + num_columns] = 0

    return image_tensor


def apply_spectral_augment(image,num):
    if num==1:
        aug_image=set_random_rows_to_zero(image, 20) #20 rows <-- frequency domain
        return aug_image
    elif num==2:
        aug_image=set_random_columns_to_zero(image, 30) #30 columns <--time domain
        return aug_image
    else:
        aug_image1=set_random_rows_to_zero(image, 20)
        aug_image=set_random_columns_to_zero(aug_image1, 30)
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
    waveform,_ = torchaudio.load(audio_path)
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    transform = transforms.Normalize(mean=[mel_spec_db.mean()], std=[mel_spec_db.std()+1e-8]),


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
    transform =transforms.Normalize(mean=[mel_spec_db.mean()], std=[mel_spec_db.std()+1e-8]), # Clamp standard deviation

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


import tqdm
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


def test_one_epoch(model,test_loader,hparams):
    test_acc=0
    device= hparams['device']
    model.eval()
    total_num_correct=0
    for i, (feats, labels) in enumerate(test_loader):
        features=feats.unsqueeze(1).to(device)
        pred = test_file(features,model)
        #print(f"If {pred} and {torch.squeeze(labels)}, then {pred==torch.squeeze(labels)}")
        if pred==torch.squeeze(labels):
            total_num_correct+=1
    return total_num_correct/i




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
    trainstage2_file = pd.read_csv(csv_file_stage2)
    train_dataset = AudioDataset(csv_file=file[file['split']=='train'],device=device,transform=compute_mel_spectrogram_aug)
    train_dataset_stage2 = AudioDataset(csv_file=trainstage2_file[trainstage2_file['split']=='train'],device=device,transform=compute_mel_spectrogram)

    valid_dataset = AudioDataset(csv_file=file[file['split']=='valid'].reset_index(drop=True),device=device,transform=compute_mel_spectrogram)
    test_dataset = AudioDataset(csv_file=test_file,device=device,transform=compute_mel_spectrogram)
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False,num_workers=4,sampler=DistributedSampler(train_dataset),drop_last=True)
    train_loader_stage2 = DataLoader(train_dataset_stage2, batch_size=hparams['batch_size'], shuffle=False,num_workers=4,sampler=DistributedSampler(train_dataset_stage2),drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=hparams['batch_size'], shuffle=False,drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False,drop_last=False)

    torch.set_float32_matmul_precision('high')

    model_slow = resnet50(num_classes=1, in_channels=1)
    
    if hparams['load_pretrained']==True:
        state_dict = torch.load(hparams['pretrained_model_path']) # this should be loaded to CUDA:0 by default and but it does not matter since we transfer it to 'rank' device later on.
        model_slow.load_state_dict(state_dict)
    
    model_slow =model_slow.to(rank)
    

    
    device=rank

    if device==0:
        print(f"Number of parameters of model is:{sum(p.numel() for p in model_slow.parameters())}")

    ddp_mp_model = torch.compile(model_slow)
    ddp_mp_model = DDP(ddp_mp_model, device_ids=[rank])
    
    # Optimizer config
    optimizer = optim.Adam(ddp_mp_model.parameters(), lr=3e-4)
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
            torch.save(ckp, f'checkpoints/new_gen_{epoch+1}_trial1.pt')
            print(f'Model saved for checkpoint:{epoch+1} at location checkpoints/new_gen_{epoch+1}_trial2.pt')
            print(f'Train loss at epoch:{epoch+1} is: {train_loss}')
            print(f'Valid loss at epoch:{epoch+1} is: {valid_loss}')
            print('---------------------------------/-------------------------------------')

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
            torch.save(ckp, f'checkpoints/new_gen_{hparams["num_stage_1_epochs"]+epoch}_trial2.pt')
            print(f'Model saved for checkpoint:{hparams["num_stage_1_epochs"]+epoch} at location checkpoints/new_gen_{hparams["num_stage_1_epochs"]+epoch}_trial1.pt')
            print(f'Train loss at epoch:{hparams["num_stage_1_epochs"]+epoch} of second stage is: {train_loss_stage2}')
            print(f'Valid loss at epoch:{hparams["num_stage_1_epochs"]+epoch} of second stage is: {valid_loss}')
            print('---------------------------------/-------------------------------------')



def main(rank: int, world_size: int, hparams: dict):
    ddp_setup(rank,world_size)
    trainer(rank,world_size,hparams)
    destroy_process_group()

if __name__=="__main__":
    import sys
    world_size  = torch.cuda.device_count()
    hparams={
    'train_csv_file':'',
    'train_csv_file_stage2': '',
    'test_csv_file': '',
    'batch_size':256,
    'num_stage_1_epochs':10,
    'num_stage_2_epochs':2,
    'num_devices': world_size,
    'load_pretrained': True,
    'pretrained_model_path': '',
    'clip_norm': 1.0,
    }
    
    mp.spawn(main,args=(world_size,hparams),nprocs=world_size)
