import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

class FairModel:
    def __init__(self, model=None, groups=None, tolerance=0.05):
        self.model = model if model is not None else LogisticRegression()
        self.groups = groups if groups is not None else {}
        self.tolerance = tolerance
        self.predictions = None

    def train(self, input, y):
        y = (y > np.median(y)).astype(int)

        self.model.fit(input, y)
        self.predictions = self.model.predict_proba(input)[:, 1]
        self._adjust_predictions(input, y)

    def _adjust_predictions(self, input, y):
        for group_name, group_indices in self.groups.items():
            group_preds = self.predictions[group_indices]
            group_labels = y[group_indices]

            prediction_error = np.abs(np.mean(group_preds) - np.mean(group_labels))

            if prediction_error > self.tolerance:
                correction = np.sign(np.mean(group_preds) - np.mean(group_labels)) * self.tolerance
                self.predictions[group_indices] -= correction
                self.predictions = np.clip(self.predictions, 0, 1)

    def predict(self, input):
        return (self.predictions > 0.5).astype(int)

    def load_data_from_csv(self, file_path, target_column, group_column=None):
        data = pd.read_csv(file_path)
        
        # Drop non-numeric columns except target and group columns
        numeric_data = data.select_dtypes(include=[np.number])

        # Ensure the target column is included
        if target_column not in numeric_data:
            numeric_data[target_column] = data[target_column]

        # Handle missing values by replacing NaN with the column mean
        imputer = SimpleImputer(strategy='mean')
        numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

        # Separate features and target
        y = numeric_data_imputed[target_column].values
        input = numeric_data_imputed.drop(columns=[target_column]).values

        # Create groups if group_column is provided
        groups = {}
        if group_column is not None and group_column in data:
            # checks if data is empty and drops the row if empty
            for group_value in data[group_column].dropna().unique():
                group_indices = np.where(data[group_column] == group_value)[0]
                groups[f"Group_{group_value}"] = group_indices
        
        return input, y, groups

if __name__ == "__main__":
    file_path = 'world_population.csv'
    target_column = '1990 Population'
    group_column = 'Continent'
    
    # Load data from the CSV
    model = FairModel()
    input, y, groups = model.load_data_from_csv(file_path, target_column, group_column)

    # Assign groups and train the model
    model.groups = groups
    model.train(input, y)

    # Make predictions
    # y = binary values (if population is above the median, it’s 1, otherwise 0).
    predictions = model.predict(input)
    print("\nAdjusted Predictions:", predictions)