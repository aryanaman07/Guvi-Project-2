import streamlit as st

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬")

import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

@st.cache_resource
def load_model():
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    model = joblib.load("logreg_sentiment_model.joblib")
    return tfidf, model

tfidf, model = load_model()

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below and find out if it's Positive or Negative.")

user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        sentiment = "Positive" if pred == 1 else "Negative"

        if sentiment == "Positive":
            st.success(f"✅ Predicted Sentiment: **{sentiment}**")
        else:
            st.error(f"❌ Predicted Sentiment: **{sentiment}**")

        st.write(f"Confidence - Positive: {prob[1]*100:.2f}%, Negative: {prob[0]*100:.2f}%")



