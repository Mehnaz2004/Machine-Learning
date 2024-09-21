import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

dataset= pd.read_csv(r"Classification\Logistic-Regression\Binomial\tested.csv")
dataset["Age"]= dataset['Age'].fillna(dataset['Age'].mean())
print((dataset.isnull().sum()/dataset.shape[0])*100)

ohe=OneHotEncoder()
dataset["Sex"]=ohe.fit_transform(dataset[["Sex"]]).toarray()
print(dataset.head(3))

#multiple inputs that is Sex and Age so we check if it linear
sns.scatterplot(x="Age", y='Sex', data=dataset, hue="Survived")
plt.show()
dataset.drop(columns=["Pclass"], inplace=True)

x=dataset.iloc[:,:-1]
y=dataset["Survived"]

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
lr=LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test)*100)
print(lr.predict([[1,47]]))

cf = confusion_matrix(y_test,lr.predict(x_test))

sns.heatmap(cf,annot=True)
plt.show()


#ALL THESE VALUES SHOULD BE HIGH
print(precision_score(y_test, lr.predict(x_test))*100)

print(recall_score(y_test, lr.predict(x_test))*100)

print(f1_score(y_test, lr.predict(x_test))*100)
