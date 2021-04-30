""" Credit by
Nama	: Muhammad Rizal Bagus Prakasa
NIM		: 081911633071
Hari,Tgl: Thursday, 29 April 2021
GitHub	: https://github.com/ubeann
"""

#%% import library 
import pandas as pd     # Pandas, docs: https://pandas.pydata.org/docs/

#%% import modules
from sklearn.preprocessing import LabelEncoder  # docs: https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
from sklearn.naive_bayes import GaussianNB      # docs: https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
from collections import defaultdict             # docs: https://www.kite.com/python/docs/collections.defaultdict

#%% import dataset
data = pd.read_excel('DataSet.xlsx', sheet_name='Data')

#%% prepare labeling dataset because data type is String
encoder_dict = defaultdict(LabelEncoder)                                              # Creating dict for labeling data (Optional)
inverse_transform_lambda = lambda x: encoder_dict[x.name].inverse_transform(x)        # Creating command for inverse labeling (Optional)

#%% choosing coloumns for features & labels
features:list = ['Harga Tanah (C1)', 'Jarak dari pusat kota (C2)', 'Ada angkutan umum (C3)']
labels:list   = ['Dipilih untuk perumahan (C4)']

#%% slicing dataset to x (features) & y (labels)
x = data[features]
y_training = data.loc[:9, labels]

#%% labeling dataset
x = x.apply(lambda x: encoder_dict[x.name].fit_transform(x))
y_training = y_training.apply(lambda x: encoder_dict[x.name].fit_transform(x))

#%% slicing variable x to x_training & x_testing
x_training = x[:10]
x_testing  = x[10:11]

#%% modeling naive bayes case using GaussianNaiveBayes
model = GaussianNB()

#%% training model
model.fit(x_training, y_training)

#%% predict using data testing
result = model.predict(x_testing)

#%% print data training (features)
print('Data Training (Features):')
data.loc[:9, ['Aturan ke-']].join(x_training.apply(inverse_transform_lambda))

#%% print data training (labels)
print('Data Training (Labels):')
data.loc[:9, ['Aturan ke-']].join(y_training.apply(inverse_transform_lambda))

#%% print data testing (features)
print('Data Testing (Features):')
x_testing.apply(inverse_transform_lambda).reset_index(drop=True)

#%% print data testing (features)
result = pd.DataFrame(data={'Dipilih untuk perumahan (C4)':[result[0]]}).apply(inverse_transform_lambda)
print('Result Data Testing (Labels):', result.loc[0, 'Dipilih untuk perumahan (C4)'])

#%% print conclusions
print('Jadi daerah dengan harga tanah MAHAL, jarak dari pusat kota SEDANG, dan ADA angkutan umum, maka daerah ini "', result.loc[0, 'Dipilih untuk perumahan (C4)'], '" layak dijadikan area perumahan.', sep='')