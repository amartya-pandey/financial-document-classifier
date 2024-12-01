# Financial Document Classifier

## Project Overview
The **Financial Document Classifier** is a machine learning-based tool designed to help finance analysts automatically classify financial documents into categories such as financial reports, market analysis, investment summaries, and more. By leveraging NLP and machine learning techniques, the classifier processes various document types (e.g., PDF, DOCX, PPTX, TXT) to streamline the categorization of financial data, enabling analysts to focus on more strategic tasks.

### Key Features:
- **Document Classification**: Classifies financial documents into predefined categories.
- **Data Preprocessing**: Text cleaning, tokenization, stopword removal, stemming, and n-gram extraction.
- **Model Training**: Decision Tree Classifier is used to train the model on preprocessed data.
- **Model Evaluation**: Includes evaluation metrics like accuracy, precision, recall, F1-score, and confusion matrix visualization.
- **Web Interface**: An easy-to-use Flask web app for uploading and classifying documents.

## Features
- **Document Types Supported**: 
    - PDFs
    - Word Documents (DOCX)
    - PowerPoint Presentations (PPTX)
    - Plain Text Files (TXT)
- **Preprocessing**: Tokenization, stopword removal, stemming, and TF-IDF vectorization.
- **Modeling**: Decision Tree classifier for document categorization.
- **Evaluation**: Model performance evaluation using cross-validation, confusion matrices, and feature importance plotting.
- **Web Interface**: A simple web interface to upload and classify financial documents.

## Installation Instructions

### Clone the Repository:
```bash
git clone https://github.com/amartya51/Financial-classifier.git
cd financial-document-classifier
```

### Install Dependencies:

Make sure you have `pip` installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies:
- `Flask`: Web framework for building the interface.
- `nltk`, `scikit-learn`: For text preprocessing, machine learning, and evaluation.
- `PyPDF2`, `python-docx`, `python-pptx`: For reading PDFs, DOCX, and PPTX files.
- `dill`: For serializing and deserializing the trained model.
- `matplotlib`, `seaborn`: For data visualization (confusion matrix, feature importance, etc.).

## Usage

### Training the Model

Open the `model.ipynb` file in notebook.
```bash
jupyter notebook
```
Do cell-wise execution of `model.ipynb` to load, preprocess data, train the model, and evaluate its performance.

This will:
- Load data from the specified directory (`data/`).
- Preprocess text data (cleaning, tokenizing, removing stopwords, etc.).
- Train the Decision Tree Classifier.
- Evaluate the model's performance using cross-validation and generate visualizations (confusion matrix, feature importance).

### Running the Web Application
To run the Flask-based web application and start classifying documents, use the following command:

```bash
python app.py
```

Then, open your web browser and go to `http://127.0.0.1:5000/`. You will be able to upload documents and classify them into categories.

## Model Details

The classifier uses the following techniques and models:

- **Preprocessing**:
  - **Tokenization**: Split text into words.
  - **Stopword Removal**: Remove common words that do not contribute to meaning.
  - **Stemming**: Reduce words to their root form (e.g., "running" -> "run").
  - **Vectorization**: TF-IDF or CountVectorizer for feature extraction from text.

- **Model**: 
  - **Classifier**: Decision Tree Classifier.
  - **Cross-Validation**: Stratified K-Fold Cross-Validation to evaluate performance.

- **Evaluation Metrics**:
  - **Accuracy**: Percentage of correctly predicted documents.
  - **Precision, Recall, F1-Score**: For detailed classification performance.
  - **Confusion Matrix**: Visual representation of model performance across classes.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

```

### Notes:
-There may be some bugs and abnormailities in this code. Please contact if you notice any of these and want to help me.
```