Following is the detailed desription of construction of Keyword spotter module with Kaldi framework.

First step is to clone the kaldi directory using: git clone https://github.com/kaldi-asr/kaldi.git

Now, follow the instructions from the repo to build Kaldi WITH cuda. We need cuda for training phase.

After kaldi is built, enter /kaldi/egs/snips/v1/ and examine readme file. To run the pipeline, modify run.sh file. In notebook generate_metadata_kaldi.ipynb one can find suitable formatting for constructing dataset.

Extracted data will be saved in /data/ directory where all the augmentations will also be positioned. 

To complete the pipeline, the required time is around 3 days. 
