import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
dataset_url = 'https://example.com/path/to/your/dataset.csv'  # Replace with your dataset URL
data = pd.read_csv(dataset_url)

# Preview the data
print(data.head())

# Data Preprocessing (example)
# Fill missing values or normalize features if necessary
data.fillna(data.mean(), inplace=True)  # Simple imputation

# Split the data into features and labels
X = data.drop(columns=['target_column'])  # Replace 'target_column' with your label
y = data['target_column']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to NumPy arrays for PyTorch
X_train = X_train.values.astype(np.float32)
y_train = y_train.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)
y_test = y_test.values.astype(np.float32)
