import pandas as pd

dataset=pd.read_csv("Cross Validation\linear-reg.csv")
x=dataset.iloc[:,:-1]
y=dataset["package"]

new_data=dataset.head(10)
x_new=new_data.iloc[::,-1]
y_new=new_data["package"]
from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold, StratifiedKFold

#lo=LeaveOneOut()
lo=KFold(n_splits=5)#baaki waale bhi aise hi kuch toh karenge


for train, test in lo.split(x_new,y_new):
    print(train,test)