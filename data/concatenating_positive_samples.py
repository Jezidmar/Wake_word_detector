#(1) Concatenating 
import os
import librosa
import soundfile as sf
import numpy as np
import random
import tqdm
def create_additional_wavs(directory, output_dir, sample_rate=16000, extra_percentage=0.2):
    # List all .wav files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    # Calculate the number of additional concatenated files to create (20% of the original file count)
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
            data, sr = librosa.load(filepath, sr=sample_rate)
            audio_data.append(data)
        
        # Concatenate along the time axis (axis=0)
        concatenated_data = np.concatenate(audio_data, axis=0)
        
        # Save the concatenated audio
        output_filename = f'extra_concatenated_{i}_clean.wav'
        output_filepath = os.path.join(output_dir, output_filename)
        sf.write(output_filepath, concatenated_data[:56000], sample_rate)
        #print(f"Extra concatenated file {output_filename} created with {len(files_to_concatenate)} files.")

# Example usage
directory = '/mnt/nvme_ssd_3/Data_v3_nc/positive_samples_aug'
output_dir = '/mnt/nvme_ssd_3/Data_v3_nc/positive_samples_aug'
create_additional_wavs(directory, output_dir)