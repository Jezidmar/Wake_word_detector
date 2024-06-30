# Construction of Keyword Spotter Module with Kaldi Framework

1. Clone the Kaldi repository:
   ```bash
git clone https://github.com/kaldi-asr/kaldi.git

3. Follow the instructions from the repo to build Kaldi WITH cuda. We need cuda for training phase.

4. After Kaldi is built, enter `/kaldi/egs/snips/v1/` and examine readme file.

5. To run the pipeline, modify run.sh file.

6. In notebook `generate_metadata_kaldi.ipynb` one can find suitable formatting for constructing dataset.

7. Extracted data will be saved in `/data/` directory where all the augmentations will also be positioned.

8. To complete the pipeline, the required time is around 3 days.
