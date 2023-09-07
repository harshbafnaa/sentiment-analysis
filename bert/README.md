# Gujarati Sentiment Analysis Using Indic Bert

This project demonstrates sentiment analysis for Gujarati text using the Indic Bert model. It includes a Streamlit web application for real-time sentiment analysis and a trained model for Gujarati text sentiment classification.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Web Application](#web-application)
- [License](#license)

## Getting Started

These instructions will help you set up and run the project on your local machine.

1. Clone the repository:

   ```bash
   git clone https://github.com/harshbafnaa/sentiment-analysis.git
   cd bert
   ```

2. Create a Python virtual environment and install the required packages:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Download the pre-trained Indic Bert model and Gujarati stopwords as mentioned in the code.

## Usage

1. Train the model: If you want to retrain the sentiment analysis model with your own dataset, follow the instructions in `model.py`. 

2. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

   This will start the web application locally.

3. Enter Gujarati text into the web application and click the "Analyze Sentiment" button to get the sentiment prediction.

## Dataset

The sentiment analysis model was trained on a dataset of Gujarati text with sentiment labels. You can replace it with your own dataset for customization.

## Model

The trained model is saved in the `model/indic_bert_model.pkl` file. You can use this model for sentiment analysis tasks related to Gujarati text.

## Web Application

The Streamlit web application in `app.py` allows you to perform real-time sentiment analysis on Gujarati text. It uses the pre-trained model for predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits
This project was created by [Harsh Bafna](https://github.com/harshbafnaa).

This project is licensed under the MIT License.
```
