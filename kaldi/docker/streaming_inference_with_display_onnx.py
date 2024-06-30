import numpy as np
import sounddevice as sd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from loading_model import load_model
from generating_file_name import generate_filename
import soundfile as sf
import argparse
import numpy as np
import cv2
import librosa
from activation_function import Sigmoid
from initialize_feats_extractors import initialize_mel_spectrogram_non_chunked_transform, initialize_mel_spectrogram_chunked_transform
import torch

# Replace NormalizeTransform with MinMaxNormalizeTransform in the pipeline

def extract_mel_combined_window(input):
    S_torch = mel_transform_entire_flow( torch.tensor(input) )
    S_DB_litorch = librosa.power_to_db(S_torch, ref=np.max)
    S_normalized = (S_DB_litorch - np.min(S_DB_litorch)) / (np.max(S_DB_litorch) - np.min(S_DB_litorch) + 1e-8)
    S_resized = cv2.resize(S_normalized, dsize=(128, 200), interpolation=cv2.INTER_LINEAR)
    return S_resized

def process_window(session, window, threshold):
    S = mel_transform( torch.tensor(window) )
    S = np.expand_dims(S, axis=0)
    S = np.expand_dims(S, axis=0)
    output=run_inference(session, S)
    return Sigmoid(output[0][0][0])

def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio processing parameters")
    parser.add_argument("--model_chunked_path", type=str, help="Path to chunked model")
    parser.add_argument("--model_non_chunked_path", type=str, help="Path to non-chunked model")
    parser.add_argument("--step_size", type=int, default=1600, help="Step size for audio processing")
    parser.add_argument("--save_path",type=str, help="Path where to save processed samples")
    args = parser.parse_args()
    return args



def run_inference(session, input_data):
    # Get the name of the first input of the model
    input_name = session.get_inputs()[0].name
    
    # Perform inference
    outputs = session.run(None, {input_name: input_data})
    
    return outputs


def audio_callback(indata, frames, time_info, status):
    global buffer, global_index, flag, record, first,wake_word_count,record_probs
    if status:
        print("\033[KStatus:", status)  # Clear to the end of line before printing status

    audio = np.reshape(indata[:, 0], -1)
    buffer = np.concatenate((buffer, audio))

    while len(buffer) >= window_size:
        window = buffer[:window_size]
        output_probability = process_window(session, window, threshold)
        
        # Display Resnet model probability and status
        print("\033[7;0H\033[K", end='')  # Move cursor to row 7 and clear line
        wakeword_status = "ACTIVE" if output_probability > threshold else "INACTIVE"
        print(f"                 Se_res2net     {output_probability:.3f}   {wakeword_status}")
        
        if output_probability > threshold:
            flag = 1
            if first == 1:
                record.append(window)
                first = 0
                record_probs.append(output_probability)
            else:
                record.append(buffer[window_size - step_size:window_size])
                record_probs.append(output_probability)

            # Handle the detection and combined probability
        else:
            if flag == 1:
                combined_window = np.concatenate(record)
                feats = extract_mel_combined_window(combined_window)
                feats = np.expand_dims(feats, axis=0)
                feats = np.expand_dims(feats, axis=0)
                output = run_inference(session_nc, feats)
                combined_window_prob = Sigmoid(output[0][0][0])
                chunks_prob = np.mean(record_probs)
                Total_probs = (combined_window_prob+chunks_prob)/2
                if Total_probs > 0.5:
                    # Clear and update the branch line
                    print("\033[8;0H\033[K", end='')  # Move cursor to row 8 and clear line
                    print("                     |")
                    # Clear and update the combined probability line
                    print("\033[9;0H\033[K", end='')  # Move cursor to row 9 and clear line
                    print("                     |___ Combined window probability: {:.3f} ---- {} detected! Total count: {}".format(Total_probs, "WAKE WORD", wake_word_count+1))
                else:
                    # Clear and update the branch line
                    print("\033[8;0H\033[K", end='')  # Move cursor to row 8 and clear line
                    print("                     |")
                    # Clear and update the combined probability line
                    print("\033[9;0H\033[K", end='')  # Move cursor to row 9 and clear line
                    print("                     |___ Combined window probability: {:.3f} ---- {} detected! Total count: {}".format(Total_probs, "NOT", wake_word_count))
   
                filename = generate_filename(prefix="sample", extension=".wav")
                #
                rounded_prob = "{:.2f}".format(Total_probs)
                if Total_probs>0.5:
                    sf.write(args.save_path +rounded_prob+'_TRUE_'+ filename, combined_window, samplerate=16000)
                    wake_word_count+=1
                else:
                    sf.write(args.save_path+rounded_prob+'_FALSE_' + filename, combined_window, samplerate=16000)
                
                flag = 0
                record = []
                record_probs = []
                first = 1

        buffer = buffer[step_size:]



args = parse_arguments()
sample_rate = 16000
window_size = int(1.2 * sample_rate)
step_size = args.step_size
buffer = np.zeros(window_size, dtype=np.float32)
global_index = 0
record = []
record_probs = []
flag = 0
first = 1
wake_word_count=0
# Load interpreters and model details
session = load_model(args.model_chunked_path)
session_nc = load_model(args.model_non_chunked_path)
threshold = 0.4
mel_transform = initialize_mel_spectrogram_chunked_transform() #Initialize mel spectrogram transform
mel_transform_entire_flow = initialize_mel_spectrogram_non_chunked_transform()


# Setup and run the streaming
with sd.InputStream(callback=audio_callback, dtype='float32', channels=1, samplerate=sample_rate):
    os.system('clear')
    print("#"*100)
    print("Listening for wakewords...")
    print("#"*100)
    output_string_header = """
            Model Name         | Score | Recording Status
            -------------------------------------------------
            """
    print(output_string_header)
    #

    try:
        sd.sleep(int(1e6))  # Run for a long time (or handle with a proper exit condition)
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")