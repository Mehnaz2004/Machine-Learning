import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Regression\Linear-Regression\linear-reg.csv")

#converting the features into 2 dimensional arrays
x=dataset[["cgpa"]]
y=dataset[["package"]]
plt.figure(figsize=(5,3))
sns.scatterplot(x="cgpa", y="package", data=dataset)
plt.show()#yeh hai nahi in slant line

#train-testsplit
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)#ideal no. of random state 42
#linear regression
lr= LinearRegression()
lr.fit(x_train, y_train)#model has been trained

print(lr.predict([[8.16]])) #making it predict something

print(lr.score(x_test, y_test)*100)#its predicting capability

print(lr.coef_)#finding value of m in y=mx+c
print(lr.intercept_) #value of c in y=mx+c

#how to know where the line of linear regression is
plt.figure(figsize=(5,4))
sns.scatterplot(x="cgpa", y="package", data=dataset)
plt.plot(dataset["cgpa"], lr.predict(x))
plt.legend(["org", "predict line"])#making menu
plt.savefig("predict.jpg")#saving the graph
