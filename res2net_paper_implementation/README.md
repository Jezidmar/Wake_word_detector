Following are the details on implementation of paper: https://arxiv.org/abs/2209.15296

In summary, system architecture is below. 
![Alt text](images/arch.png)

Work is entirely based on synthesized data. One can find scripts for synthesizing data in folder /data. For now, only TortoiseTTS and WaveGlow are used for synthetizing. In future work, I will add VCTK and MARS. 

For capturing voice embeddings, I used the following datasets:
  - VoxCeleb Gender: https://dagshub.com/DagsHub/audio-datasets/src/main/voice_gender_detection (~7000 voices)
  - Speech accent archive: https://www.kaggle.com/datasets/rtatman/speech-accent-archive (~2300 voices)
  - People's Speech: https://mlcommons.org/datasets/peoples-speech/ (~15000 voices / randomly selected)


For training each model, I used roughly 3M audio files from People's speech dataset and chunk of CommonVoice dataset for the negative samples. This is approximately 5500h of speech. Production grade systems are trained on multiple times bigger datasets so I expect the achieved performance to be increased by further expanding the training set size. As for positive samples, I used following augmentation pipeline:  
<img src="images/aug_pipeline.png" alt="Alt text" width="300"/>
After the model is trained, we can convert it into onnx or tflite format suitable for edge device. Find .ipynb script for converting in /models folder.


