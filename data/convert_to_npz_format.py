import os
import numpy as np
import tqdm
import librosa
import cv2
import random
def compute_mel_spectrogram(audio_path,   n_mels=128, sr=16000):
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
    y, sr = librosa.load(audio_path, sr=sr)
    #if len(y) < 2048:
    #    print(f"Warning: '{audio_path}' File is not large enough.")
    # Compute the Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=1024,hop_length=160,  n_mels=n_mels)
    
    # Convert to log scale (dB)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_normalized = (S_DB - np.min(S_DB)) / (np.max(S_DB) - np.min(S_DB)+1e-8) #clamp
    S_resized = cv2.resize(S_normalized, dsize=(128,200), interpolation=cv2.INTER_LINEAR)

    #print(S_normalized.shape)
    return S_resized

def compute_mel_spectrogram_aug(audio_path,   n_mels=128, sr=16000):
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
    y, sr = librosa.load(audio_path, sr=sr)
    # if len(y) < 2048:
    #    print(f"Warning: '{audio_path}' File is not large enough.")
    #Compute the Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr,hop_length=160,n_fft=1024,  n_mels=n_mels)
    
    # Convert to log scale (dB)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_normalized = (S_DB - np.min(S_DB)) / (np.max(S_DB) - np.min(S_DB)+1e-8) #clamp

    num = random.randint(1,3)
    aug_mel = apply_spectral_augment(S_normalized,num)
    
    S_resized = cv2.resize(aug_mel, dsize=(128,200), interpolation=cv2.INTER_LINEAR)
    
    return S_resized






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

def apply_spectral_augment(image,num):
    if num == 1:
        aug_image = set_random_rows_to_zero(image, 15)  # 20 rows in the frequency domain
        return aug_image
    elif num == 2:
        aug_image = set_random_columns_to_zero(image, 20)  # 30 columns in the time domain
        return aug_image
    else:
        aug_image1 = set_random_rows_to_zero(image, 15)
        aug_image = set_random_columns_to_zero(aug_image1, 20)
        return aug_image

base_directory = '/mnt/nvme_ssd_3/Data_v3_nc'
def create_individual_npz_files(base_directory):
    categories = {
      #  'positive_samples': 1,
      #  'positive_samples_1':1,
      #  'positive_samples_aug':1,
        'negative_samples': 0,
        'negative_samples_1':0,
    }
    
    for category, label in categories.items():
        directory_path = os.path.join(base_directory, category)
        files = os.listdir(directory_path)
        print(directory_path)
        #print(files)
        for file in tqdm.tqdm(files):
            if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
                file_path = os.path.join(directory_path, file)
                try:
                    feats = compute_mel_spectrogram(file_path)
                    feats_aug = compute_mel_spectrogram_aug(file_path)
                    output_path = '/mnt/nvme_ssd_1/Data_v3_nc_npz/'+str(category)+'/'+file[:50] +'.npz'
                    np.savez_compressed(output_path, feats=feats,feats_aug=feats_aug, label=label)
                except:
                    continue;
                
create_individual_npz_files(base_directory)
