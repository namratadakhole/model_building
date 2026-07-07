import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load the trained model and TF-IDF vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vector.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text


def preprocess_text_for_model(text):
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmas]
    stemmed_text = ' '.join(stemmed_tokens)
    return stemmed_text


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
