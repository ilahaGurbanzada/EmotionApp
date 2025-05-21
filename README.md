# Emotion Detection App (Streamlit + HuggingFace Token)

This app detects emotions in short texts like tweets or comments using a fine-tuned BERT model on the GoEmotions dataset.

## How to Deploy

1. Upload this project to a GitHub repository
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New app", select the repo and path to `app.py`
4. In your app's **Settings > Secrets**, add:

```
HF_TOKEN = your_real_token_here
```

Your Hugging Face token must have "read" access (generate it at https://huggingface.co/settings/tokens)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
