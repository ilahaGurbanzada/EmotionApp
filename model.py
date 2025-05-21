
import torch
import torch.nn as nn
from transformers import BertTokenizer
import pickle

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

def load_model(model_name="LSTM"):
    if model_name == "LSTM":
        with open("outputs/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        label_map = {0: "joy", 1: "anger", 2: "sadness", 3: "fear", 4: "neutral"}
        model = torch.load("outputs/lstm_model.pt", map_location=torch.device("cpu"))
        model.eval()
        return model, tokenizer, label_map
    else:
        return None, None, None

def predict_emotion(text, model, tokenizer, label_map):
    tokens = tokenizer.texts_to_sequences([text])
    input_tensor = torch.tensor(tokens, dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = torch.argmax(outputs, dim=1).item()
    return label_map[prediction]
