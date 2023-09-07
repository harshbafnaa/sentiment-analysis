import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import nltk
import os

# Function to load custom Gujarati stopwords from a text file
def load_gujarati_stopwords(stopwords_file):
    stopwords = []
    if stopwords_file is not None:
        stopwords = stopwords_file.read().decode('utf-8').splitlines()
    return stopwords

# Function to preprocess text by removing stopwords
def preprocess_text(text, stopwords):
    if isinstance(text, str):
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in stopwords and word.isalnum()]
        return ' '.join(filtered_tokens)
    else:
        return ""

# Load your dataset (replace 'your_dataset.csv' with your dataset file)
dataset = pd.read_csv('dataset/dataset.csv')

# Drop rows with NaN values in the 'Text' or 'Label' columns
dataset.dropna(subset=['Text', 'Label'], inplace=True)

# Load custom Gujarati stopwords (replace 'your_stopwords.txt' with your stopwords file)
stopwords_file = open('stopwords/gujarati_stopwords.txt', 'rb')  # Replace with your stopwords file
gujarati_stopwords = load_gujarati_stopwords(stopwords_file)

# Preprocess the 'Text' column
dataset['Text'] = dataset['Text'].apply(lambda x: preprocess_text(x, gujarati_stopwords))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['Text'], dataset['Label'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a sentiment analysis model (e.g., Naive Bayes)
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model to a .pkl file
joblib.dump(clf, 'model/sentiment_analysis_model.pkl')

# Save the TF-IDF vectorizer to a .pkl file
joblib.dump(tfidf_vectorizer, 'model/tfidf_vectorizer.pkl')

# Close the stopwords file
stopwords_file.close()
