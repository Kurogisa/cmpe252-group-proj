import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class FairModel:
    def __init__(self, model=None, groups=None, tolerance=0.05):
        # checks if model is provided 
        if model is not None:
            self.model = model
        else:
            self.model = LogisticRegression()

        # checks if group is provided
        if groups is not None:
            self.groups = groups
        else:
            self.groups = {}

        # Tolerance to adjust the accuracy difference between groups
        self.tolerance = tolerance

        # Store predictions here
        self.predictions = None

    def train(self, input, y):
        """
        Train the model on the dataset.
        """
        self.model.fit(input, y)
        self.predictions = self.model.predict_proba(input)[:, 1]
        self._adjust_predictions(input, y)

    def _adjust_predictions(self, input, y):
        """
        Adjust predictions to ensure fairness for each group.
        """
        for group_name, group_indices in self.groups.items():
            group_preds = self.predictions[group_indices]
            group_labels = y[group_indices]

            prediction_error = np.abs(np.mean(group_preds) - np.mean(group_labels))

            if prediction_error > self.tolerance:
                correction = np.sign(np.mean(group_preds) - np.mean(group_labels)) * self.tolerance
                self.predictions[group_indices] -= correction
                self.predictions = np.clip(self.predictions, 0, 1)

    def predict(self, input):
        """
        Make final predictions based on the adjusted probabilities.
        """
        return (self.predictions > 0.5).astype(int)

    def load_data_from_csv(self, file_path, target_column, group_column=None):
        """
        Load dataset from a CSV file, ensuring only numeric data is used.
        """
        data = pd.read_csv(file_path)
        
        # Drop non-numeric columns (except the target and group columns)
        numeric_data = data.select_dtypes(include=[np.number])

        # Ensure the target column is included
        numeric_data[target_column] = data[target_column]

        # Separate features and target
        y = numeric_data[target_column].values
        input = numeric_data.drop(columns=[target_column]).values
        
        groups = {}
        if group_column is not None:
            # Create groups dynamically
            group_values = data[group_column].unique()
            for group_value in group_values:
                group_indices = np.where(data[group_column] == group_value)[0]
                groups[f"Group_{group_value}"] = group_indices
        
        return input, y, groups

if __name__ == "__main__":
    # Example: Using CSV to load data
    file_path = 'world_population.csv'  # Replace with your CSV file path
    target_column = '1990 Population'  # Replace with the column name of your target variable
    group_column = 'Continent'  # Replace with the column name of your group column, if any
    
    # Load data from the CSV
    model = FairModel()
    input, y, groups = model.load_data_from_csv(file_path, target_column, group_column)

    # Create and train the model
    model.groups = groups
    model.train(input, y)

    # Make predictions
    predictions = model.predict(input)
    print("Adjusted Predictions:", predictions)
