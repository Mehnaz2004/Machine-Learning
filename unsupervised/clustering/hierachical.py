import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv(r"unsupervised\clustering\Iris.csv")

print(dataset.head(3))

#graph making or visualization

sns.pairplot(data=dataset)
plt.show()

#so first we will try to make a dendogram and then find the no. of clusters and then using agglomerative clustering we will get the desired answer

import scipy.cluster.hierarchy as sc

plt.figure(figsize=(20,20))
sc.dendrogram(sc.linkage(dataset, method='single', metric='euclidean'))
plt.savefig("demo.jpg")
plt.show()

from sklearn.cluster import AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters=2, linkage='single')
dataset["Predict"]=ac.fit_predict

sns.pairplot(data=dataset, hue="Predict")
plt.show()