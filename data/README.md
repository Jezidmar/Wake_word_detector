### For data augmentation run following:
```
python3 adding_background_noise_and_speech.py \
    --path_to_background_noise "$PATH_TO_BACKGROUND_NOISE" \
    --path_to_background_speech "$PATH_TO_BACKGROUND_SPEECH" \
    --path_to_audio "$PATH_TO_AUDIO" \
    --save_path "$SAVE_PATH"
```

### For offline feature extraction:
```
python3 compute_npz_files.py \
    --path_to_audio_files "$PATH_TO_AUDIO_FILES" \
    --save_path "$SAVE_PATH"
```

### For concatenating positive samples so as to create additional data:
```
python3 concatenating_positive_samples.py \
    --path_to_audio_files "$PATH_TO_AUDIO_FILES" \
    --save_path "$SAVE_PATH" \
    --extra_percentage "$EXTRA_PERCENTAGE"
```
