import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp import common
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import pickle

# Load Gujarati stopwords
stopwords = set()
with open("stopwords/gujarati_stopwords.txt", "r", encoding="utf-8") as file:
    stopwords.update(file.read().split())

# Function to clean and preprocess text
def clean_text(text):
    # Remove special characters, numbers, and extra whitespaces, while keeping Gujarati characters
    text = re.sub(r'[^઀-૿\s]', '', text)
    # Tokenize using Indic NLP library
    tokens = indic_tokenize.trivial_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

# Load your dataset
data = pd.read_csv("dataset/dataset.csv")  # Replace with your dataset file path
data = data.dropna()  # Remove rows with missing values
data['Text'] = data['Text'].apply(clean_text)  # Clean and preprocess the text

# Convert labels to integers (e.g., 0 for Non-hostile, 1 for Hostile)
data['Label'] = data['Label'].map({'Non-hostile': 0, 'Hostile': 1})

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(data['Text'], data['Label'], test_size=0.2, random_state=42)

# Load the Indic BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)

# Tokenize the text data and create DataLoader for training and validation
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Create TensorDatasets with correctly formatted labels
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels.values))
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels.values))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
for epoch in range(3):  # You can adjust the number of epochs
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation on validation set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report
report = classification_report(all_labels, all_preds)
print(report)

model_path = "model/indic_bert_model.pkl"  # Choose a suitable file path and name
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)