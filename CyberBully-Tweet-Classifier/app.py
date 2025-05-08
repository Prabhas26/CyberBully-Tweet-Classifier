import streamlit as st
import joblib
from preprocess import clean_text
from preprocess import load_data
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_resource
def load_model(binary=True):
    suffix = "binary" if binary else "multiclass"
    model = joblib.load(f"models/cyberbully_model_{suffix}.pkl")
    vectorizer = joblib.load(f"models/vectorizer_{suffix}.pkl")
    return model, vectorizer


def predict_text(tweet, binary=True):
    model, vectorizer = load_model(binary)
    cleaned = clean_text(tweet)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)
    proba = model.predict_proba(vect)
    confidence = max(proba[0])
    return pred[0], confidence



def predict_csv(file, binary=True):
    df = pd.read_csv(file)
    df['tweet_text'] = df['tweet_text'].astype(str).apply(clean_text)
    model, vectorizer = load_model(binary)
    X = vectorizer.transform(df['tweet_text'])
    preds = model.predict(X)
    df['prediction'] = preds
    return df


def evaluate_model(binary=True):
    (X_train, X_test, Y_train, Y_test), _ = (load_data('Data Set Refining/tweets_refined.csv', binary))
    model, _ = load_model(binary)
    Y_Pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_Pred)
    report = classification_report(Y_test, Y_Pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(Y_test, Y_Pred)
    return acc, report, cm


st.title("Cyberbullying Tweet Classifier")
st.warning("Note: This model is trained on historical data and may reflect biases. Predictions should not be used for "
           "punitive actions without human review.")
st.header("Classify tweets as Cyberbullying or Not (Binary / Multi-Class)")

mode = st.radio("Select Mode", ["Binary Classification", "Multi-Class Classification"])
binary_mode = (mode == "Binary Classification")

menu = ["Classify Single Tweet", "Classify from CSV", "Evaluate Model"]
choice = st.sidebar.selectbox("Select Action", menu)

if choice == 'Classify Single Tweet':
    tweet = st.text_area("Enter the tweet here:")
    if st.button("Classify"):
        result, confidence = predict_text(tweet, binary=binary_mode)
        if binary_mode:
            label = "Cyberbullying" if result == 1 else "Not Cyberbullying"
            st.success(f"Prediction:{label}")
        else:
            st.success(f"Cyberbullying Type: {result}")
            st.info(f"Prediction Confidence:{confidence:.2f}")
elif choice == "Classify from CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with a 'tweet_text' column", type=["csv"])
    if uploaded_file is not None:
        output_df = predict_csv(uploaded_file, binary=binary_mode)
        st.dataframe(output_df)
        st.download_button("Download Predictions",
                           output_df.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")
elif choice == "Evaluate Model":
    st.write("Evaluating Model...")
    acc, report, cm = evaluate_model(binary=binary_mode)
    st.metric(label="Accuracy", value=f"{acc * 100:.2f}%")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)
