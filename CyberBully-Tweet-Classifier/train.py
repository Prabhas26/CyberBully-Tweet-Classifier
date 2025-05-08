import os
import joblib
from sklearn.linear_model import LogisticRegression
from preprocess import load_data


def train_model(binary=True):
    (X_train, X_test, Y_train, Y_test), vectorizer = load_data('Data Set Refining/tweets_refined.csv', binary)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, Y_train)

    suffix = "binary" if binary else "multiclass"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/cyberbully_model_{suffix}.pkl")
    joblib.dump(vectorizer, f"models/vectorizer_{suffix}.pkl")
    print(f"{suffix.capitalize()} model and vectorizer saved!")


if __name__ == "__main__":
    train_model(binary=True)
    train_model(binary=False)
