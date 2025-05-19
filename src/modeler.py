from sklearn.ensemble import RandomForestClassifier


class Modeler:
    def __init__(self, model=None):
        # default to a simple RF classifier
        self.model = model or RandomForestClassifier(
            n_estimators=100, random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):


    def predict(self, X_new):
        return self.model.predict(X_new)
