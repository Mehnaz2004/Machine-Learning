import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv(r"Regression\Multiple Linear Reg\multiple_linear_regression_dataset.csv")

print(dataset.isnull().sum())#no empty values

#making graphs to check if they are linear
sns.pairplot(data=dataset)
plt.show()

#so experience with income is linear but not age and so we can also see this in other way
sns.heatmap(data=dataset.corr(), annot=True)
plt.show()

x=dataset.iloc[:,:-1]
y=dataset['income']

#train-test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state = 42)

Mr= LinearRegression()
Mr.fit(x_train, y_train)

print(Mr.score(x_test, y_test)*100)
print(Mr.predict([[43,10]]))