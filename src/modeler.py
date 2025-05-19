from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

class Modeler:
    def __init__(self, model=None):
        # default to a simple RF classifier
        self.model = model or RandomForestClassifier(
            n_estimators=100, random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:,1]
        print("Classification Report:")
        print(classification_report(y_test, preds))
        print(f"ROC AUC: {roc_auc_score(y_test, probs):.3f}")

    def predict(self, X_new):
        return self.model.predict(X_new)
