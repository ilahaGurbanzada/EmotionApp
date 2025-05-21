import streamlit as st
from transformers import pipeline

st.title("Emotion Detection in Social Media Posts")
st.write("Enter a social media post and detect its underlying emotion.")

classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-goemotions-ekman")

text = st.text_area("Enter text", "")

if text:
    result = classifier(text)
    st.write("### Detected Emotion:")
    st.json(result)
