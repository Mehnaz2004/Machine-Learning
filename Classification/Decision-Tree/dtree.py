import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv(r"Social_Network_Ads.csv")

print(dataset.head(3))

x=dataset.iloc[:,:-1]
y=dataset['Purchased']

#SCALING KARLO AS THE VALUES HAVE A BIG DIFFERENCE
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
sc.fit(x)
sc.transform(x)

x=pd.DataFrame(sc.transform(x), columns=x.columns)

print(x.head(3))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(max_depth=5) #pre-pruning ka tarika
dt.fit(x_train,y_train)

print(dt.score(x_test,y_test)*100)

print(dt.predict([[19,19000]]))

#how to graph analysis

from sklearn.tree import plot_tree
plt.figure(figsize=(20,20))
plot_tree(dt)
plt.show()

print(dt.score(x_train,y_train)*100)

#post pruning is nothing but ki finding the best value of max_depth required and then using that to prune the data then
#this is done using a loop over various values of max depth over a range and seeing which value of max_depth has the least difference betwen the dt scores of train and test 