### For exporting trained module to .onnx file:
```python3
python3 convert_M0_to_onnx.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --save_model_path "$SAVE_MODEL_PATH" \
    --use_version_1 "$USE_VERSION_1"
```

Only difference between M0 and M1 versions is the input image dimension.
