import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vector.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Lowercase
    text = text.lower()

    # Remove stopwords
    text = " ".join(
        [word for word in text.split() if word not in stop_words]
    )

    return text


def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    stems = [stemmer.stem(token) for token in lemmas]
    return " ".join(stems)


# ---------------- Streamlit UI ----------------

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered",
)

st.title("😊 Sentiment Analysis App")

st.write("Enter a review below and click Predict.")

review = st.text_area("Enter Review")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter some text.")
    else:

        processed = preprocess_text(review)

        vector = vectorizer.transform([processed])

        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")
