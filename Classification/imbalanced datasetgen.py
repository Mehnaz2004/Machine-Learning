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

# Generate imbalanced random binary labels (90% class 0 and 10% class 1)
proportion_of_class_1 = 0.1  # 10% of the samples will be class 1
num_class_1 = int(proportion_of_class_1 * num_samples)  # Number of class 1 samples
num_class_0 = num_samples - num_class_1  # Number of class 0 samples

# Create labels (90% 0s, 10% 1s)
random_labels = np.array([0] * num_class_0 + [1] * num_class_1)

# Shuffle the labels to avoid ordering
np.random.shuffle(random_labels)

# Ensure labels are either 0 or 1
assert set(random_labels).issubset({0, 1}), "Labels are not binary!"

# Create a DataFrame for features and imbalanced labels
data = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
data['Label'] = random_labels

# Check for missing values (there should be none)
assert not data.isnull().values.any(), "There are missing values!"

# Save the dataset to CSV without index
data.to_csv('Classification\imb.csv', index=False)

print("Imbalanced dataset created, rounded off, verified, and saved as 'imb.csv'")
