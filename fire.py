
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pickle

import numpy as np



def randomforest(xtrain,xtest,ytrain,ytest):
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(xtrain,ytrain) #training

    
    with open('mlmodellr.pickle','wb') as f: #saving model
        pickle.dump(clf,f)
    
    pkl = open('mlmodellr.pickle', 'rb')
    clf = pickle.load(pkl)   

    y_pred=clf.predict(xtest)
    
    acc=clf.score(xtest,ytest)    
    
    return acc*100


df=pd.read_csv('forestfire.csv')

print(df.columns)


print(df.head())
df.dropna(inplace=True) #pre-processing

label_encoder = preprocessing.LabelEncoder()
df['target']= label_encoder.fit_transform(df['target'])


unique = df['target'].unique()
y=df['target']


print("unique values:",unique)



y = df['target']
df.drop('target', inplace=True, axis=1)


x = df
print(x)



xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)


accrf=randomforest(xtrain,xtest,ytrain,ytest)


print(accrf)



    
    


