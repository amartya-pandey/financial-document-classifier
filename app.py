import os
import dill
import pandas as pd
from flask import Flask, request, render_template
import chardet  # To detect file encoding
from io import BytesIO
from PyPDF2 import PdfReader  # For PDF files
from docx import Document  # For DOCX files

app = Flask(__name__)

# Load the model and vectorizer
with open('model.dill', 'rb') as model_file:
    model = dill.load(model_file)

with open('vectorizer.dill', 'rb') as vectorizer_file:
    vectorizer = dill.load(vectorizer_file)

# Define the categories
categories = [
    "Consolidated statement of cash flows",
    "Statement of changes in equity",
    "Note to financial statements",
    "Statement of operations",
    "Statements of Financial Position"
]

def read_file_content(file):
    # Read the raw data from the file
    raw_data = file.read()
    
    # Detect the encoding
    result = chardet.detect(raw_data)
    encoding = result['encoding'] if result['encoding'] is not None else 'utf-8'  # Default to utf-8

    # Reset file pointer to the beginning
    file.seek(0)

    # Try to read as text
    try:
        content = raw_data.decode(encoding)
    except UnicodeDecodeError:
        # If decoding fails, try to handle specific formats
        if file.filename.endswith('.pdf'):
            reader = PdfReader(BytesIO(raw_data))
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""
        elif file.filename.endswith('.docx'):
            doc = Document(BytesIO(raw_data))
            content = "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError("Unsupported file format or encoding.")
    
    return content

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            try:
                # Read the file content
                content = read_file_content(file)
                # Vectorize the content
                vectorized_content = vectorizer.transform([content])
                
                # Make prediction (single category)
                prediction_index = model.predict(vectorized_content)[0]  # Get the first (and only) prediction
                
                # Map the prediction index to the category
                predicted_category = categories[prediction_index]  # Assuming the model outputs an index
                
                return render_template('index.html', result=predicted_category)
            except Exception as e:
                return render_template('index.html', result=f"Error: {str(e)}")
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)