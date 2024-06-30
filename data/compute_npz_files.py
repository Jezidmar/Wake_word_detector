import os
import numpy as np
import tqdm
import librosa
import cv2
base_directory = '/mnt/nvme_ssd_3/Data_v3_nc'
def create_individual_npz_files(base_directory, output_directory):
    categories = {
        'positive_samples': 1,
        'positive_samples_1':1,
        'positive_samples_aug':1,
        'negative_samples': 0,
        'negative_samples_1':0,
    }
    
    for category, label in categories.items():
        directory_path = os.path.join(base_directory, category)
        print('1')
        files = os.listdir(directory_path)
        print(directory_path)
        #print(files)
        for file in tqdm.tqdm(files):
            if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
                file_path = os.path.join(directory_path, file)
                feats = compute_mel_spectrogram(file_path)
                output_path = '/mnt/nvme_ssd_3/Data_v3_nc_npz/'+str(category)+'/'+file +'.npz'
                np.savez_compressed(output_path, feats=feats, label=label)
                
create_individual_npz_files(base_directory,output_file)
