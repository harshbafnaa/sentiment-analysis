import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained sentiment analysis model
clf = joblib.load('model/sentiment_analysis_model.pkl')  # Replace with the path to your model

# Load the fitted TF-IDF vectorizer (replace 'tfidf_vectorizer.pkl' with your fitted vectorizer file)
tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')  # Replace with the path to your fitted vectorizer

# Function to preprocess text
def preprocess_text(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.isalnum()]
    return ' '.join(filtered_tokens)

# Streamlit app
st.title("Gujarati Tweet Sentiment Analysis")
st.sidebar.subheader("Enter a Gujarati Sentence:")

user_input = st.sidebar.text_area("Input your sentence:")

if st.sidebar.button("Analyze Sentiment"):
    user_input_preprocessed = preprocess_text(user_input)
    user_input_tfidf = tfidf_vectorizer.transform([user_input_preprocessed])
    prediction = clf.predict(user_input_tfidf)[0]

    st.subheader("Sentiment Analysis Result:")
    if prediction == 'Hostile':
        st.error("Hostile")
    else:
        st.success("Non-hostile")
