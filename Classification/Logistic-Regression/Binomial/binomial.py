import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

dataset= pd.read_csv(r"Classification\Logistic-Regression\Binomial\tested.csv")
dataset["Age"]= dataset['Age'].fillna(dataset['Age'].mean())
dataset.drop(columns=["Sex", "Pclass"], inplace=True)
print((dataset.isnull().sum()/dataset.shape[0])*100)

sns.scatterplot(x="Age", y="Survived", data=dataset)
plt.show()

x=dataset[["Age"]]
y=dataset[["Survived"]]

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
lr=LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test)*100)#this table should have been used for more than one inputs lr but for learning purpose we used it here aswell