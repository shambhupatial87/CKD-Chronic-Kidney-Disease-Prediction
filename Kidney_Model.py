import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("kidney.csv")

df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']] = df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']].replace(to_replace={'\t8400':'8400', '\t6200':'6200', '\t43':'43', '\t?':np.nan, '\tyes':'yes', '\tno':'no', 'ckd\t':'ckd'})

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

df[['pcv', 'wc', 'rc']] = df[['pcv', 'wc', 'rc']].astype('float64')

df.drop(['id', 'sg', 'pcv', 'pot'],axis=1,inplace=True)

col = ['rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane']
encoder = LabelEncoder()
for col in col:
    df[col] = encoder.fit_transform(df[col])
    
df[['appet', 'classification']] = df[['appet', 'classification']].replace(to_replace={'good':'1', 'ckd':'1', 'notckd':'0', 'poor':'0'})
df[['classification', 'appet']] = df[['classification', 'appet']].astype('int64')
    
X = df.drop("classification", axis=1)
y = df["classification"]

scaler = StandardScaler()
features = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42) 

dTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11)
dTree.fit(x_train, y_train)

import pickle
pickle.dump(dTree, open('model_dtree.pkl','wb'))
model_dtree = pickle.load(open('model_dtree.pkl','rb'))
