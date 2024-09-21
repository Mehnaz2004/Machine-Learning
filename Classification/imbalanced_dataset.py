import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

dataset= pd.read_csv(r"Classification\imb.csv")

x=dataset.iloc[:,:-1]
y=dataset["Label"]

#RANDOM UNDER SAMPLER
print(dataset["Label"].value_counts())
ru=RandomUnderSampler()
ru_x, ru_y = ru.fit_resample(x,y)
print(ru_y.value_counts())#IT WORKSS GIVES 47.5 SCORE


#RANDOM OVER SAMPLING
ro=RandomOverSampler()
ro_x, ro_y = ro.fit_resample(x,y)
print(ro_y.value_counts())#IT WORKSS GIVES 54.164 SCORE


x_train, x_test, y_train, y_test=train_test_split(ro_x,ro_y, test_size=0.2, random_state=42)
lr=LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test)*100)