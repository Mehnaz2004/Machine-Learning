import pandas as pd
from sklearn.model_selection import train_test_split
dataset = pd.read_csv("cleaned_data.csv")

input_data = dataset.iloc[:, [0, 1, 2, 6, 7, 8]]
output_data = dataset.iloc[:, [3,4,5]]

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.25)
