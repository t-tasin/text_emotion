# Text Emotion Detection

## Overview
This repository contains the code and models for a Text Emotion Detection project, capable of identifying various emotions such as anger, disgust, fear, happiness, joy, neutrality, sadness, shame, and surprise from textual data. It leverages machine learning algorithms to analyze and predict the emotional tone conveyed in sentences or phrases.

## Try the App
If you're interested in trying out the text emotion detection app without setting up the environment, visit the hosted version at [Text Emotion Analyzer](https://textemotion-analyzer.streamlit.app/).

## Structure
- `Dataset/`: Contains the datasets used for training the models.
  - `text.csv`: Dataset for model training.
  - `tweet_emotions.csv`: Additional dataset for extended model training.
- `Model/`: Stores the trained machine learning models.
  - `text_emotion.pkl`: Trained model for emotion detection.
  - `text_emotion_6param.pkl`: An alternate model trained on a subset of emotions.
- `Train_Model/`: Jupyter notebooks used for training the models.
  - `Model1_6Param.ipynb`: Notebook for training the model with six parameters.
  - `Model2.ipynb`: Notebook for training a more comprehensive model.
- `web_app.py`: A Streamlit web application that serves the trained models for real-time emotion detection in text.

## Features
- Emotion detection in text using logistic regression and other algorithms.
- Visualization of prediction probabilities.
- Streamlit web application for interacting with the models.

## Usage
To run the Streamlit web application:
1. Ensure you have Streamlit installed in your environment.
2. Navigate to the repository's root directory.
3. Execute `streamlit run web_app.py` in your terminal.
4. The web application will start, and you can interact with it through your browser.

## Requirements
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Altair
- Joblib
- Scikit-learn
- Neattext

Install the necessary Python packages using `pip install -r requirements.txt` (ensure you have a `requirements.txt` file that lists these packages).

## Contributing
Contributions to improve the models or the application are welcome. Please fork the repository, make your changes, and submit a pull request.

---
