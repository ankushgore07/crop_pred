# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 05:38:44 2021

@author: Ankush
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import missingno as msno
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#importing train test split for spliting data
from sklearn.model_selection import train_test_split


#importing dataset

ds= pd.read_csv(r'C:\Users\Ankush\Desktop\myproject\Crop_recommendation.csv')


X= ds.iloc[:,0:7]

y=ds.label

#converting the labels into numeric value
#le=LabelEncoder()
#y=le.fit_transform(y)

X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=40)
rc= RandomForestClassifier()

rc.fit(X_train,y_train)
sc=rc.score(X_test,y_test)
print(sc)

result=rc.predict([[90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362]])
print(result)

#saving model to disk
pickle.dump(rc,open('model.pkl','wb'))

#loding model to compare rsult
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362]])) 



