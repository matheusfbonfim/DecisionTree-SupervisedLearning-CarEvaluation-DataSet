import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np

# read .csv from provided dataset
csv_filename="car.data"

# df=pd.read_csv(csv_filename,index_col=0)
df=pd.read_csv(csv_filename,
              names=["Buying", "Maintenance" , "Doors" , "Persons" , "Lug-Boot" , "Safety", "Class"])


df.head()


#Convert car-class labels to numbers
le = preprocessing.LabelEncoder()
df['Class'] = le.fit_transform(df.Class)

df['Class'].unique()

array([2, 0, 3, 1], dtype=int64)


features = list(df.columns)
features.remove('Class')