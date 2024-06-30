import os
os.environ["HF_HOME"] = "/mnt/nvme_ssd_1/marin_data"
os.environ["HF_DATASETS_CACHE"] = "/mnt/nvme_ssd_1/marin_data"

from huggingface_hub import login 
login(' -- code --')

from datasets import load_dataset

dataset = load_dataset("MLCommons/peoples_speech",split="Train",name='dirty')
