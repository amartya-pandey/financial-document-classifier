from flask import Flask, request, render_template
import dill
import os

# Load the model and vectorizer
with open("model.dill", "rb") as model_file:
    model = dill.load(model_file)

with open("vectorizer.dill", "rb") as vectorizer_file:
    vectorizer = dill.load(vectorizer_file)

# Flask app initialization
app = Flask(__name__)

# Categories for reference
categories = [
    "Consolidated statement of cash flows",
    "Note to financial statements",
    "Statement of changes in equity",
    "Statement of operations",
    "Statements of Financial Position"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["file"]
        if file:
            try:
                # Try reading as UTF-8
                content = file.read().decode("utf-8")
            except UnicodeDecodeError:
                # If UTF-8 fails, fallback to ISO-8859-1 (Latin-1)
                file.seek(0)  # Reset file pointer
                content = file.read().decode("iso-8859-1")

            # Transform the content using the vectorizer
            transformed_content = vectorizer.transform([content])
            # Predict the category
            pred = model.predict(transformed_content)[0]
            prediction = categories[pred]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
