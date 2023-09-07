import streamlit as st
import pickle
from transformers import AutoTokenizer
import torch

# Load the pre-trained model
model_path = "model/indic_bert_model.pkl"  # Replace with the path to your .pkl file
with open(model_path, 'rb') as model_file:
   model = pickle.load(model_file)

# Load the Indic BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

# Streamlit app
st.title("Gujarati Sentiment Analysis Using Indic Bert")
text_input = st.text_area("Enter Gujarati Text")

if st.button("Analyze Sentiment"):
   if text_input:
       # Tokenize and preprocess the input text
       tokens = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=128)
       input_ids = tokens["input_ids"]
       attention_mask = tokens["attention_mask"]

       # Make a prediction
       with torch.no_grad():
           output = model(input_ids, attention_mask=attention_mask)
           prediction = torch.argmax(output.logits, dim=1).item()

       # Display the sentiment prediction
       sentiment = "Hostile" if prediction == 1 else "Non-hostile"
       st.write(f"Predicted Sentiment: {sentiment}")
   else:
       st.warning("Please enter some text for analysis.")
