import streamlit as st
import os
from transformers import pipeline

st.title("Emotion Detection in Social Media Posts")

hf_token = os.getenv("HF_TOKEN")

classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-goemotions-ekman",
    use_auth_token=hf_token
)

text = st.text_area("Enter a tweet or comment:")
if text:
    result = classifier(text)
    st.write("### Detected Emotion:")
    st.json(result)
