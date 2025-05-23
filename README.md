# Cyber Bully Tweet Classifier

This project is an AI-based web application built with Streamlit that classifies tweets as either cyberbullying or not. It supports both *binary* and *multi-class* classification modes.

## Features

- Clean and preprocess tweet text
- Train Logistic Regression models for binary & multi-class classification
- Evaluate model performance (accuracy, classification report, confusion matrix)
- Streamlit interface for:
  - Classifying single tweets
  - Uploading CSVs for batch prediction
  - Evaluating model accuracy

## Installation

### Clone the repo:
```bash
git clone https://github.com/Prabhas26/CyberBully-Tweet-Classifier.git
cd CyberBully-Tweet-Classifier
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Dataset:
Place the tweets_refined.csv dataset inside a folder named Data Set Refining/.

## How to Run the App

### Train the models:
open the folder in the IDE you use to run the project and enter the following command to train models.
```
python train.py
```

### Run the Streamlit app:
```
streamlit run app.py --server.headless=true
```
Then open the URL shown in the terminal in a browser.



