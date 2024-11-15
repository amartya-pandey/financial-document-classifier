# bootstrapping.py

class Bootstrapping:
    def __init__(self, model):
        self.model = model

    def retrain_on_misclassified(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        misclassified_indices = [i for i, (pred, true) in enumerate(zip(predictions, y_test)) if pred != true]
        
        if misclassified_indices:
            X_misclassified = X_test[misclassified_indices]
            y_misclassified = y_test[misclassified_indices]
            self.model.fit(X_misclassified, y_misclassified)
            print("Model retrained on misclassified articles.")
