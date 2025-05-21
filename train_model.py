
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from utils import get_goemotions, preprocess_text
from model import SimpleLSTM
import pickle
import os
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.tokenizer.texts_to_sequences([self.texts[idx]])[0]
        seq = pad_sequences([seq], maxlen=self.max_len)[0]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def train():
    data = get_goemotions()
    texts = [preprocess_text(x['text']) for x in data]
    labels = [x['labels'][0] if x['labels'] else 0 for x in data]  # use first label or 0

    label_counts = Counter(labels)
    top_labels = [label for label, _ in label_counts.most_common(5)]
    filtered_texts = [t for t, l in zip(texts, labels) if l in top_labels]
    filtered_labels = [top_labels.index(l) for l in labels if l in top_labels]

    x_train, x_val, y_train, y_val = train_test_split(filtered_texts, filtered_labels, test_size=0.2)

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    max_len = 50

    train_dataset = EmotionDataset(x_train, y_train, tokenizer, max_len)
    val_dataset = EmotionDataset(x_val, y_val, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SimpleLSTM(vocab_size=5000, embedding_dim=64, hidden_dim=64, output_dim=5)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss / len(train_loader)}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model, "outputs/lstm_model.pt")
    with open("outputs/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Model and tokenizer saved!")

if __name__ == "__main__":
    train()
