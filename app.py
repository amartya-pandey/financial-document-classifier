from flask import Flask, request, jsonify
import pandas as pd
from data_loader import DataLoader
from preprocessing import TextPreprocessor
from vectorization import Vectorizer
from model_manager import load_model

app = Flask(__name__)

# Load model and vectorizer
model_filename = 'model.pkl'
model = load_model(model_filename)
vectorizer = Vectorizer()

# Endpoint for predicting categories of new documents
@app.route('/asdf', methods=['POST'])
def predict():
    # Check if JSON data is sent
    if not request.is_json:
        return jsonify({"error": "Invalid input format. JSON expected."}), 400

    # Parse the JSON data
    data = request.get_json()
    documents = data.get('documents', None)

    # Check if 'documents' key exists in the JSON
    if documents is None or not isinstance(documents, list):
        return jsonify({"error": "Invalid input. A list of documents is expected under the key 'documents'."}), 400

    # Preprocess the documents
    preprocessor = TextPreprocessor()
    processed_documents = preprocessor.preprocess_documents(documents)
    text_data = [' '.join(tokens) for filename, tokens in processed_documents]  # Reconstruct text from tokens

    # Vectorize the text data
    X = vectorizer.transform(text_data)

    # Predict categories
    predictions = model.model.predict(X)

    # Prepare the output as JSON
    response = [
        {"filename": processed_documents[i][0], "category": predictions[i]}
        for i in range(len(predictions))
    ]

    return jsonify(response)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

if __name__ == "__main__":
    app.run(debug=True)
