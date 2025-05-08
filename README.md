# Financial Document Classifier

## Project Overview
The **Financial Document Classifier** is a machine learning-based tool designed to help finance analysts automatically classify financial documents into categories such as financial reports, market analysis, investment summaries, and more. By leveraging NLP and machine learning techniques, the classifier processes various document types (e.g., PDF, PPTX, TXT) to streamline the categorization of financial data, enabling analysts to focus on more strategic tasks.

### Key Features:
- **Document Classification**: Classifies financial documents into predefined categories.
- **Data Preprocessing**: Text cleaning, tokenization, stopword removal, stemming, and n-gram extraction.
- **Model Training**: Decision Tree Classifier is used to train the model on preprocessed data.
- **Self-Bootstrapping**: The model improves iteratively based on user feedback.
- **Model Evaluation**: Includes evaluation metrics like accuracy, precision, recall, F1-score, and confusion matrix visualization.
- **Web Interface**: An easy-to-use Flask web app for uploading and classifying documents.

## Features
- **Document Types Supported**: 
    - PDFs
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
- `PyPDF2`, `python-pptx`: For reading PDFs, DOCX, and PPTX files.
- `dill`: For serializing and deserializing the trained model.
- `matplotlib`, `seaborn`: For data visualization (confusion matrix, feature importance, etc.).

## Usage

### Training the Model
Open the `model.ipynb` file in a notebook.
```bash
jupyter notebook
```
Execute cells in `model.ipynb` to load, preprocess data, train the model, and evaluate its performance.

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

## Self-Bootstrapping Process
The classifier incorporates a self-bootstrapping mechanism based on user feedback:

1. **Prediction and Feedback**: After the document is classified, the user can provide feedback on whether the prediction was correct. 
2. **Feedback Storage**: The feedback is saved in a `feedback.csv` file for further analysis.
3. **Retraining the Model**: The model automatically retrains itself using the updated data, incorporating new user feedback to improve accuracy.
4. **Model Persistence**: The updated model and vectorizer are saved and used for subsequent predictions.

### Example Workflow:
- A user uploads a document and receives a prediction.
- If the prediction is incorrect, the user can select the correct category.
- The feedback is stored, and the model retrains, enabling continuous improvement.

---

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

---
## Contributors
- **[Amartya51](https://github.com/amartya51)**: Lead developer and maintainer.
- **[Naman Singh](https://github.com/namansinghr)**: Contributions in improving model performance and feedback mechanisms.
- **[Shreya Pandey](https://github.com/Shreya1393)**: Contributions in improving model performance and feedback mechanisms.
- **[Aditi Rai](https://github.com/whoaditi)**: Contributions in improving model performance and feedback mechanisms.
---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

### Notes:
There may be some bugs and abnormalities in this code. Please contact if you notice any issues and want to help improve the project.
