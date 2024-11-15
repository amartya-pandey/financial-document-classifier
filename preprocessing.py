# preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        return text

    def tokenize(self, text):
        tokens = text.split()
        return [self.stem(token) for token in tokens if token not in self.stop_words]

    def stem(self, word):
        return self.stemmer.stem(word)

    def preprocess_documents(self, documents):
        processed_docs = []
        for filename, text in documents:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize(cleaned_text)
            processed_docs.append((filename, tokens))
        return processed_docs
