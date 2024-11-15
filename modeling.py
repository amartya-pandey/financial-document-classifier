# modeling.py

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class Model:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        print(classification_report(self.y_test, predictions))

    def tune_hyperparameters(self, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters found: ", grid_search.best_params_)
        return grid_search.best_estimator_
 