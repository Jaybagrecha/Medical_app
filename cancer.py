import pandas as pd
import numpy as np
data = pd.read_csv('cancer.csv')
import pickle
#from sklearn.externals import joblib

data.drop(["Unnamed: 32"],axis="columns",inplace=True)
data.drop(["id"],axis="columns",inplace=True)

a=pd.get_dummies(data["diagnosis"])

cancer=pd.concat([data,a],axis="columns")

cancer.drop(["diagnosis","B"],axis="columns",inplace=True)

cancer.rename(columns={"M":"Malignant/Benign"},inplace=True)

y=cancer[["Malignant/Benign"]]
X=cancer.drop(["Malignant/Benign"],axis="columns")

from xgboost import XGBClassifier
classifierX = XGBClassifier(gamma=0.3,learning_rate=0.3,n_estimators=100,reg_alpha=0,reg_lambda=1)

X=np.array(X)
y=np.array(y)
classifierX.fit(X,y.reshape(-1,))
#joblib.dump(classifierX,"model")
pickle.dump(classifierX,open('model_cancer.pkl','wb'))
