import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import streamlit as st
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold


#Importing dataset

dataset  = pd.read_csv('Expresso_churn_dataset.csv')

dataset.info()

data = dataset.select_dtypes(include=['number'])
#Specify the threshold for variance
threshold = 0.01  # You can adjust this threshold as needed

#Initialize the VarianceThreshold transformer
selector = VarianceThreshold(threshold)

#Fit and transform the data
data_reduced = selector.fit_transform(data)
#Get a boolean mask of selected features
selected_features_mask = selector.get_support()

#Get the names of the selected features
selected_features = data.columns[selected_features_mask]
data_reduced_df = pd.DataFrame(data_reduced, columns=selected_features)

#DETERMINER LE NOMBRE DES NAN
missing_values = data_reduced_df.isnull().sum()
print("Missing values:\n", missing_values)

# replace nan with mode dans une colonne

data_reduced_df['MONTANT'].fillna(data_reduced_df['MONTANT'].mode()[0], inplace=True)
data_reduced_df['FREQUENCE_RECH'].fillna(data_reduced_df['FREQUENCE_RECH'].mode()[0], inplace=True)
data_reduced_df['REVENUE'].fillna(data_reduced_df['REVENUE'].mode()[0], inplace=True)
data_reduced_df['ARPU_SEGMENT'].fillna(data_reduced_df['ARPU_SEGMENT'].mode()[0], inplace=True)
data_reduced_df['FREQUENCE'].fillna(data_reduced_df['FREQUENCE'].mode()[0], inplace=True)
data_reduced_df['DATA_VOLUME'].fillna(data_reduced_df['DATA_VOLUME'].mode()[0], inplace=True)
data_reduced_df['ON_NET'].fillna(data_reduced_df['ON_NET'].mode()[0], inplace=True)
data_reduced_df['ORANGE'].fillna(data_reduced_df['ORANGE'].mode()[0], inplace=True)
data_reduced_df['TIGO'].fillna(data_reduced_df['TIGO'].mode()[0], inplace=True)
data_reduced_df['ZONE1'].fillna(data_reduced_df['ZONE1'].mode()[0], inplace=True)
data_reduced_df['ZONE2'].fillna(data_reduced_df['ZONE2'].mode()[0], inplace=True)

data_reduced_df['FREQ_TOP_PACK'].fillna(data_reduced_df['FREQ_TOP_PACK'].mode()[0], inplace=True)

missing_values = data_reduced_df.isnull().sum()
print("Missing values:\n", missing_values)



# handle with duplicate , missing values
data_reduced_df = data_reduced_df.drop_duplicates()

X = data_reduced_df.iloc[:, :-1].values    # All columns but the last
y =data_reduced_df.iloc[:, -1].values    # The Last COLUMN (Purchased)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 40, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)








