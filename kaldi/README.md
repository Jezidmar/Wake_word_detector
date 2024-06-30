# Construction of Keyword Spotter Module with Kaldi Framework

## Step 1: Clone Kaldi Repository
```bash
git clone https://github.com/kaldi-asr/kaldi.git

2. Follow the instructions from the repo to build Kaldi WITH cuda. We need cuda for training phase.

3. After Kaldi is built, enter `/kaldi/egs/snips/v1/` and examine readme file.

4. To run the pipeline, modify run.sh file.

5. In notebook `generate_metadata_kaldi.ipynb` one can find suitable formatting for constructing dataset.

6. Extracted data will be saved in `/data/` directory where all the augmentations will also be positioned.

7. To complete the pipeline, the required time is around 3 days.
