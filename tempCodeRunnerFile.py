import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load the model and vectorizer
with open('model.dill', 'rb') as model_file:
    model = dill.load(model_file)

with open('vectorizer.dill', 'rb') as vectorizer_file:
    vectorizer = dill.load(vectorizer_file)
