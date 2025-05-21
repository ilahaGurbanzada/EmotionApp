
import streamlit as st
from model import load_model, predict_emotion
from utils import preprocess_text
import pandas as pd

st.title("Emotion Detection in Social Media Posts")

user_input = st.text_area("Enter your text here", "")

model_choice = st.selectbox("Choose Model", ["LSTM", "BERT (soon)"])

if st.button("Analyze"):
    if user_input:
        model, tokenizer, label_map = load_model(model_choice)
        text = preprocess_text(user_input)
        prediction = predict_emotion(text, model, tokenizer, label_map)
        st.success(f"Predicted Emotion: {prediction}")
    else:
        st.warning("Please enter some text!")
