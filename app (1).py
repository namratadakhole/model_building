from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available (download if not present)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
# We'll use the .pkl files for consistency with the last request
model = joblib.load('model.pkl') # or use pickle.load(open('model.pkl', 'rb'))
vectorizer = joblib.load('vector.pkl') # or use pickle.load(open('vector.pkl', 'rb'))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def clean_text(text):
    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # 2. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Convert to lowercase
    text = text.lower()
    # 4. Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_text_for_model(text):
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmas]
    stemmed_text = ' '.join(stemmed_tokens)
    return stemmed_text

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No text provided in the request body'}), 400

        input_text = data['text']

        # Preprocess the input text using the same steps as training
        processed_text = preprocess_text_for_model(input_text)

        # Transform the processed text using the loaded TF-IDF vectorizer
        text_vectorized = vectorizer.transform([processed_text])

        # Predict sentiment
        prediction = model.predict(text_vectorized)
        sentiment = 'positive' if prediction[0] == 1 else 'negative'

        return jsonify({'text': input_text, 'predicted_sentiment': sentiment})

if __name__ == '__main__':
    # To run the Flask app locally for testing, use:
    # app.run(debug=True, host='0.0.0.0', port=5000)
    # For Colab, you might need ngrok or a similar tool for external access.
    # For production, use a production-ready WSGI server like Gunicorn.
    print("To run this Flask app, execute: python app.py")
    print("You might need to install Flask first: pip install Flask")
    print("Then, you can send POST requests to /predict with JSON body {'text': 'your review'}")
