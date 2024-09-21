import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

dataset= pd.read_csv(r"Classification\Logistic-Regression\Binomial\lrp.csv")

sns.scatterplot(x="Feature_1", y="Feature_2", data=dataset, hue="Label")
#plt.show()#this shows that the dataset is not linear but polynomial

x=dataset.iloc[:,:-1]
y=dataset["Label"]

#firstway
pf=PolynomialFeatures(degree=0)#degree adjust karke we can increase the accuracy
pf.fit(x)

x = pd.DataFrame(pf.transform(x))

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
lr=LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test)*100)