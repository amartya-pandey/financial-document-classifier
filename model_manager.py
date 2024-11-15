import pickle

def save_model(model, filename):
    """Save the model to a file using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load a model from a file using pickle."""
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model
 