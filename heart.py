import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

h=pd.read_csv("heart.csv")
x=h.iloc[:,:-1]
y=h.iloc[:,-1]
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe column
##print(featureScores)
##print(featureScores.nlargest(10,'Score'))

x=x.drop(columns=["fbs","restecg"])

StandardScaler = StandardScaler()
column_to_scale = ['age','sex','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']
x[column_to_scale] = StandardScaler.fit_transform(h[column_to_scale])

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=.70,random_state=0)
#(x_train.shape)
#print(x_test.shape)

logreg=LogisticRegression()

x=np.array(x)
y=np.array(y)

logreg.fit(x,y.reshape(-1,))


#accuracy=logreg.score(x_test, y_test)
#print("Accuracy: {}".format(accuracy))
#predictions = logreg.predict(x_test)
#print("Confusion Matrix:")
#cm=confusion_matrix(y_test, predictions)
#print(cm)
#print("Classification Report")
#print(classification_report(y_test, predictions))
#accuracies = cross_val_score(estimator=logreg,X=x_train,y=y_train,cv=10,n_jobs=-1)
#mean = accuracies.mean()
#std = accuracies.std()
#print(cm)
#print(mean)
#print(std)

pickle.dump(logreg, open('model_heart.pkl','wb'))
