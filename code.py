""" Credit by
Nama	: Muhammad Rizal Bagus Prakasa
NIM		: 081911633071
Hari,Tgl: Saturday, 08 May 2021
GitHub	: https://github.com/ubeann
"""

#%% import library
import time                          # docs: https://docs.python.org/3/library/time.html
import numpy as np                   # docs: https://numpy.org/doc/stable/
import pandas as pd                  # docs: https://pandas.pydata.org/docs/
import matplotlib.pyplot as plt      # docs: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
from sklearn.cluster import KMeans   # docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.preprocessing import LabelEncoder  # docs: https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
from collections import defaultdict  # docs: https://www.kite.com/python/docs/collections.defaultdict

#%% import dataset
data = pd.read_csv('googleplaystore.csv')[:500:5]

#%% cleaning dataset
## check missing value
print("Missing value pada dataset:\n", data.isnull().sum(), sep='')

## drop missing value
print('\nMembersihkan missing value dengan drop row tersebut', end='')
for i in range(3):
    print('.', end='')
    time.sleep(1)
data = data.dropna().reset_index()
print("\nDataset berhasil dibersihkan\n")

## verify dataset
print("Missing value pada dataset setelah dibersihkan:\n", data.isnull().sum(), sep='')

#%% prepare labeling dataset because data type is String
encoder_dict = defaultdict(LabelEncoder)                                              # Creating dict for labeling data (Optional)
inverse_transform_lambda = lambda x: encoder_dict[x.name].inverse_transform(x)        # Creating command for inverse labeling (Optional)

#%% prepare variable
y_column = 'Genres'
x = data.loc[:, data.columns != y_column].apply(lambda x: encoder_dict[x.name].fit_transform(x)).values
y = data.loc[:,[y_column]].apply(lambda x: encoder_dict[x.name].fit_transform(x))
kmeans = KMeans(n_clusters = y.max().values[0]+1)
result = kmeans.fit_predict(x)

#%% Plot All K-Means Clusters
labels = np.unique(result)
for i in labels:
    plt.scatter(x[result == i , 0] , x[result == i , 1] , label = pd.DataFrame(data={y_column:[i]}).apply(inverse_transform_lambda).values[0][0])
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c='yellow', label = 'Centroids')
plt.title('Clusters of Apps GP')
plt.legend()
plt.show()