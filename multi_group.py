import numpy as np
from sklearn.linear_model import LogisticRegression

class MultigroupRobustModel:
    def __init__(self, base_model=None, groups=None, epsilon=0.05):
        """
        Initializes the multigroup robust model.
        :param base_model: The base learning model (e.g., Logistic Regression)
        :param groups: Dictionary mapping subgroup names to their indices in dataset
        :param epsilon: Multiaccuracy threshold
        """
        self.base_model = base_model if base_model else LogisticRegression()
        self.groups = groups if groups else {}
        self.epsilon = epsilon
        self.predictions = None

    def fit(self, X, y):
        """
        Train the base model.
        """
        self.base_model.fit(X, y)
        self.predictions = self.base_model.predict_proba(X)[:, 1]
        self._apply_multiaccuracy(X, y)

    def _apply_multiaccuracy(self, X, y):
        """
        Adjusts predictions to ensure multiaccuracy across groups.
        """
        for group, indices in self.groups.items():
            group_preds = self.predictions[indices]
            group_labels = y[indices]
            error = np.abs(np.mean(group_preds) - np.mean(group_labels))
            
            if error > self.epsilon:
                correction = np.sign(np.mean(group_preds) - np.mean(group_labels)) * self.epsilon
                self.predictions[indices] -= correction
                self.predictions = np.clip(self.predictions, 0, 1)

    def predict(self, X):
        """
        Make predictions using the adjusted probabilities.
        """
        return (self.predictions > 0.5).astype(int)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulated dataset
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Generating labels based on features
    
    # Define groups (e.g., subgroup indices)
    groups = {
        "Group_A": np.where(X[:, 0] > 0.5)[0],
        "Group_B": np.where(X[:, 0] <= 0.5)[0]
    }
    
    # Train and evaluate model
    model = MultigroupRobustModel(groups=groups)
    model.fit(X, y)
    predictions = model.predict(X)
    
    print("Adjusted Predictions:", predictions)
