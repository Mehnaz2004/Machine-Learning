import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv(r"Regression\Polynomial\salary.csv")

#you can see the graph is polynomial
plt.scatter(dataset["Level"], dataset["Salary"])
plt.show()

x=dataset[["Level"]]
y=dataset["Salary"]

#how to convert data into polynomial feature
pf=PolynomialFeatures(degree=2)
pf.fit(x)
x =pf.transform(x)

#train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#applying polynomial regression
lr=LinearRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)*100)

#making prediction line
prd = lr.predict(x)
plt.scatter(dataset["Level"], dataset["Salary"])
plt.plot(dataset["Level"], prd, y, c="red")
plt.xlabel("Level")
plt.ylabel("Salary")

plt.show()

