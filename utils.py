
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

from datasets import load_dataset

def get_goemotions():
    dataset = load_dataset("go_emotions")
    return dataset["train"]
