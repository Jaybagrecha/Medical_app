import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.externals import joblib
import matplotlib.pyplot as plt

df = pd.read_csv('liver.csv')

df=df.fillna(method="ffill")
df.Gender=df.Gender.map({"Female":1,"Male":0})
df["Dataset"]=df["Dataset"].map({1:0,2:1})

#df['Gender'].value_counts()
#df['Dataset'].value_counts()

X = df.iloc[:,:-1]
y = df.iloc[:,[10]]

# y= y.reshape(-1,)

from xgboost import XGBClassifier
classifierX = XGBClassifier()

X=np.array(X)
y=np.array(y)
classifierX.fit(X,y.reshape(-1,))
#joblib.dump(classifierX,"model")
pickle.dump(classifierX,open('model_liver.pkl','wb'))




