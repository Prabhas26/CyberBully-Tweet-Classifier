import joblib
import numpy as np
from preprocess import load_data
from sklearn.metrics import classification_report, accuracy_score


def evaluate_model(binary=True):
    (X_train, X_test, Y_train, Y_test), _ = load_data('Data Set Refining/tweets_refined.csv', binary)
    suffix = "binary" if binary else "multiclass"
    model = joblib.load(f"models/cyberbully_model_{suffix}.pkl")
    Y_Pred = model.predict(X_test)

    print(f"--- {suffix.upper()}CLASSIFICATION REPORT ---")
    print("Accuracy:", accuracy_score(Y_test, Y_Pred))
    print("Classification:\n", classification_report(Y_test, Y_Pred, zero_division=0))
    missed_labels = set(np.unique(Y_test)) - set(np.unique(Y_Pred))
    if missed_labels:
        print("Warning:The Following classes are not predicted:", missed_labels)


if __name__ == "__main__":
    evaluate_model(binary=True)
    evaluate_model(binary=False)
