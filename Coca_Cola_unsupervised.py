import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("Coca_Cola_Data.csv")
print(data)
x= data[" Annual Revenue (US dollars in Billions)"]
print(x)
data = data[["Time (Years)", "Annual Revenue (US dollars in Billions)"]]

#standardize the data
x_std = StandardScaler().fit_transform(data)

#the value of k has been defined for you
k = 5

#apply the kmeans algorithm
km = KMeans(n_clusters=k).fit(x_std) 

#get the centroid and label values
centroids = km.cluster_centers_
labels = km.labels_

#sets the size of the graph
plt.figure(figsize=(5,4))

#use a for loop to plot the data points in each cluster
for i in range(k):
    cluster = x_std[labels == i]
    print(cluster)
    plt.scatter(cluster[:,0], cluster[:,1])

#plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100,
            c='r', label='centroid')
            
#shows the graph
plt.xlabel("Time (Years)")
plt.ylabel("Annual Revenue (US dollars in Billions)")
plt.show()