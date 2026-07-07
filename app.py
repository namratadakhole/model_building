import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# ---------------------- Download NLTK Resources ----------------------

resources = {
    "corpora/stopwords": "stopwords",
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4"
}

for path, resource in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

# ---------------------- Load Model ----------------------

model = joblib.load("model.pkl")
vectorizer = joblib.load("vector.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# ---------------------- Text Cleaning ----------------------

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    text = " ".join(
        [word for word in text.split() if word not in stop_words]
    )

    return text


# ---------------------- Preprocessing ----------------------

def preprocess_text_for_model(text):
    cleaned_text = clean_text(text)

    # Tokenize
    tokens = word_tokenize(cleaned_text)

    # Lemmatize
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    # Stem
    stemmed_tokens = [stemmer.stem(token) for token in lemmas]

    return " ".join(stemmed_tokens)


# ---------------------- Streamlit UI ----------------------

st.set_page_config(
    page_title="IMDb Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and click **Predict** to determine whether the sentiment is Positive or Negative."
)

input_text = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was absolutely amazing! The acting and storyline were fantastic."
)

if st.button("Predict Sentiment"):

    if input_text.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        # Preprocess
        processed_text = preprocess_text_for_model(input_text)

        # Vectorize
        text_vectorized = vectorizer.transform([processed_text])

        # Predict
        prediction = model.predict(text_vectorized)

        sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"

        st.subheader("Prediction")
        st.success(sentiment)
