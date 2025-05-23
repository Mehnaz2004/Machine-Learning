import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv(r"Market_Basket_Optimisation.csv")

print(dataset.head(3))

#dataset has a lot of null values so first isse deal karna hoga

print(dataset["Item1"][0])

print(dataset.shape)

market=[]
for i in range(0,dataset.shape[0]):
    cus=[]
    for j in dataset.columns:
        if type(dataset[j][i])==str:
            cus.append(dataset[j][i])
    market.append(cus)
    
#print(market)

#lets get frequency of each item

l=[]

for i in market:
    for j in i:
        l.append(j)
        
import collections
p=collections.Counter(l)

d={"ItemName":p.keys(), "freq":p.values()}  #made a dataframe out of it

print(pd.DataFrame(d).sort_values(by=["freq"], ascending=False))

#true false matrix banana hai

from mlxtend.preprocessing.transactionencoder import TransactionEncoder
tr=TransactionEncoder()

tr.fit(market)

df= pd.DataFrame(tr.transform(market),columns=tr.columns_)

from mlxtend.frequent_patterns import apriori

ap=apriori(df, min_support=0.01, use_colnames=True, max_len=6).sort_values(by=["support"])  #yaha data bohot bada hai so keeping support really small because repitation is less likely, show change in max_lens and support and how it affects
print(ap)