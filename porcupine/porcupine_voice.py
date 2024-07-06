import pvporcupine
import pyaudio
import struct
import threading
import time
import queue
import matplotlib.pyplot as plt



def plot_graph(detection_list):
    time_stamps = [i*0.01 for i in range(len(detection_list))]  # Assuming detection attempt every 0.01 seconds

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(time_stamps, detection_list, drawstyle='steps-post', marker='o')
    plt.title('Wake Word Detection Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Wake Word Detected')
    plt.yticks([0, 1], ['No', 'Yes'])  # Set y-axis labels to show 'No' for 0 and 'Yes' for 1
    plt.grid(True)
    plt.show()




def audio_processor(audio_queue, porcupine, detection_list):
    while True:
        audio_frame = audio_queue.get()
        if audio_frame is None:  # Shutdown signal
            break
        # Process the audio frame
        #print(f"Len of audio frame is:{audio_frame}")
        keyword_index = porcupine.process(audio_frame)
        # Append 1 if wake word detected, else 0
        print(f'keyword_index:{keyword_index}',flush=True)
        detection_list.append(1 if keyword_index >= 0 else 0)

def main():
    porcupine = pvporcupine.create(
    access_key='FB7jo4vQwScNqXa5J2E+/nCXw3wag8lSs4oj6RpfnJCK6NZopYcxSg==',
    keyword_paths=['Hey-Jules_en_linux_v3_0_0.ppn'],
    
    )


    pa = pyaudio.PyAudio()
    audio_queue = queue.Queue()
    detection_list = []

    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    print(porcupine.frame_length)

    # Start the audio processing thread
    processor_thread = threading.Thread(target=audio_processor, args=(audio_queue, porcupine, detection_list))
    processor_thread.start()

    print("Listening for the wake word...")

    try:
        while True:
            # Read audio frame from the microphone
            audio_frame = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            audio_frame = struct.unpack_from("h" * porcupine.frame_length, audio_frame)

            # Put the frame in the queue for processing
            audio_queue.put(audio_frame)

            # Simulate lower latency by sleeping for 0.1 seconds
            #time.sleep(0.1)

            # Example: Print detection list every 0.1 seconds
            #print(detection_list[-10:])  # Print last 10 detections

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        # Clean up resources
        plot_graph(detection_list)
        
        audio_queue.put(None)  # Send shutdown signal to processor thread
        processor_thread.join()
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()








