import pandas as pd
#import numpy as np
#from sklearn.linear_model import LogisticRegression
import pickle
#from sklearn.externals import joblib
#import matplotlib.pyplot as plt

data= pd.read_csv('dataKidney.csv')

data = data.fillna(method='bfill')

y=data[["class"]].values
X=data.drop(["class"],axis="columns").values
y=y.reshape(-1,)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from xgboost import XGBClassifier
classifierX = XGBClassifier()
classifierX.fit(X_train,y_train)

y_predX = classifierX.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predX)
#print(cm)

pickle.dump(classifierX,open('model_kidney.pkl','wb'))






