# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:36:35 2020

@author: maxim
"""

import HolidaysManager as FM
import DataManager as DM
import numpy as np 


from sklearn.preprocessing import StandardScaler
from sklearn import model_selection as ms
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor


#Getting the data
def GetData():
    X = FM.FeaturesData()
    y = DM.CreateSalesFrame()
   
    y = y.drop(columns=['Year','Week Number'])
    y = y.drop([304,305,306,307,308,309,310,311])
    
    return X, y
    
    #We create training and testing data that fit with sklearn package

    X_train,X_test,y_train,y_test=ms.train_test_split( X, y, test_size=0.20, random_state=0)


    
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    #return [X_train,X_test,y_train,y_test]
    #Now We can try our models !

def KNN_Regressor():
    Data = GetData()
    X_train = Data[0]
    y_train = Data[2]
    X_test = Data[1]
    knn = KNeighborsRegressor(n_neighbors=10,n_jobs=4)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    return y_pred