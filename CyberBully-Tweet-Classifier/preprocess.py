import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()


def load_data(file_path, binary=True):
    df = pd.read_csv(file_path)
    df['tweet_text'] = df['tweet_text'].astype(str).apply(clean_text)
    if binary:
        df['label'] = df['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)
    else:
        df['label'] = df['cyberbullying_type']
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['tweet_text'])
    Y = df['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    return (X_train, X_test, Y_train, Y_test), vectorizer
