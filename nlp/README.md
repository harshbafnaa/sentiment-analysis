# Gujarati Tweet Sentiment Analysis

This is a simple Python application for performing sentiment analysis on Gujarati tweets using a Naive Bayes classifier. The project consists of two main files: `model.py`, which contains the code for training the sentiment analysis model, and `app.py`, which is a Streamlit web application for interacting with the model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/harshbafnaa/sentiment-analysis.git
   cd your-repo
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Access the web application in your browser (by default, it runs on http://localhost:8501).

3. Enter a Gujarati sentence in the sidebar text area and click the "Analyze Sentiment" button to perform sentiment analysis.

## Files and Directory Structure

- `model.py`: Contains the code for training the sentiment analysis model.
- `app.py`: Streamlit web application for interacting with the model.
- `dataset/`: Directory for the dataset (replace `your_dataset.csv` with your dataset file).
- `stopwords/`: Directory for custom Gujarati stopwords (replace `gujarati_stopwords.txt` with your stopwords file).
- `model/`: Directory for trained model and TF-IDF vectorizer.

## Credits

- [Streamlit](https://streamlit.io/): Streamlit is used to create the web application.
- [scikit-learn](https://scikit-learn.org/): scikit-learn is used for machine learning tasks.
- [NLTK](https://www.nltk.org/): NLTK is used for natural language processing tasks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

