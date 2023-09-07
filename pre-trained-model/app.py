import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit UI
st.title("Sentiment Analysis Using Pre Trained Model")

# Input text for analysis
input_text = st.text_input("Enter the text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if input_text:
        # Analyze sentiment
        sentiment_scores = analyzer.polarity_scores(input_text)

        # Determine sentiment based on the compound score
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        # Display the result
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text for sentiment analysis.")
