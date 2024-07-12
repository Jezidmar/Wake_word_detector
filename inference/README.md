### To extract DET curve:
```python3
python3 compute_det_for_M0.py \
    --test_csv_file "$TEST_CSV_FILE" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --save_path "$SAVE_PATH" \
    --use_version_1 "$USE_VERSION_1"
```
### Compute DET for Porcupine baseline:
```python3
python3 your_script_name.py \
    --test_csv_file "$TEST_CSV_FILE" \
    --save_path "$SAVE_PATH" \
    --access_key "$ACCESS_KEY" \
    --keyword_path "$KEYWORD_PATH"
```
