import os
os.environ["HF_HOME"] = ""
os.environ["HF_DATASETS_CACHE"] = ""

from huggingface_hub import login 
login(' -- code --')

from datasets import load_dataset

dataset = load_dataset("MLCommons/peoples_speech",split="Train",name='dirty')
