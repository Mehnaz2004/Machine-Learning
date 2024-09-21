import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

dataset= pd.read_csv(r"Classification\Logistic-Regression\Multinomial\Iris.csv")

print(dataset["Species"].unique())
sns.pairplot(data=dataset, hue="Species")
plt.show()
#looking at this we can calculate the feature selection, and we can see that sepal_length ahs alot of overlapping and thus it would be difficult for classification
#thus we should remove the sepal_length for feature selection but since we are only doing simple stuff we will keep it for now
x=dataset.iloc[:,:-1]
y=dataset["Species"]

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
#OVR METHOD
#lr=LogisticRegression(multi_class='ovr') #96percent
#Multinomial
#lr=LogisticRegression(multi_class='multinomial')#100percent
#AUTO
lr=LogisticRegression()#detects which one is the best to be used
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test)*100)