import numpy as np
import pandas as pd

# Parameters for dataset
np.random.seed(42)
num_samples = 1000

# Generate random features within the range of -5 to 5
X = np.random.rand(num_samples, 2) * 10 - 5  # Values between -5 and 5

# Round off feature values to 2 decimal places
X = np.round(X, 2)

# Check that all features are within the range
assert np.all((X >= -5) & (X <= 5)), "Feature values out of range!"

# Generate random binary labels (0 or 1)
random_labels = np.random.randint(0, 2, size=num_samples)

# Ensure labels are either 0 or 1
assert set(random_labels).issubset({0, 1}), "Labels are not binary!"

# Create a DataFrame for features and random labels
data = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
data['Label'] = random_labels

# Check for missing values (there should be none)
assert not data.isnull().values.any(), "There are missing values!"

# Save the dataset to CSV without index
data.to_csv('Classification\Logistic-Regression\Binomial\lrp.csv', index=False)

print("Dataset with random labels created, rounded off, verified, and saved as 'lrp.csv'")
