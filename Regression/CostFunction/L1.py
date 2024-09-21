import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge

dataset = pd.read_csv('CostFunction\linear-reg.csv')

plt.figure(figsize=(10,10))
sns.heatmap(data=dataset.corr(), annot=True)
plt.show()

#separating dependent and independent variables
x=dataset.iloc[:,:-1]
y=dataset["package"]

sc=StandardScaler()#scaled the data
sc.fit(x)

x=pd.DataFrame(sc.transform(x), columns=x.columns)
print(x.head(5))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test, y_test)*100)#u can see its really bad like -6 lol

plt.figure(figsize=(15,5))
plt.bar(x.columns, lr.coef_)
plt.show()

#LASSO
la=Lasso(alpha=0.000000001)
la.fit(x_train,y_train)
print(la.score(x_test,y_test)*100)

#Ridge
ri=Ridge(alpha=0.000000001)
ri.fit(x_train,y_train)
print(ri.score(x_test,y_test)*100)

