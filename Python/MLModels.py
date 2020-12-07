# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:36:35 2020

@author: maxim
"""

import HolidaysManager as FM
import DataManager as DM
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn import model_selection as ms

from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor




#Getting the data
def GetData():
    X = FM.FeaturesData()
    y = DM.CreateSalesFrame()
   
    y = y.drop(columns=['Year','Week Number'])
    y = y.drop([304,305,306,307,308,309,310,311])
    
    
    
    #We create training and testing data that fit with sklearn package

    X_train,X_test,y_train,y_test=ms.train_test_split( X, y, test_size=0.20, random_state=0)


    
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    return  X_train,X_test,y_train,y_test

    #return [X_train,X_test,y_train,y_test]
    #Now We can try our models !

def KNN_Regressor():
    X_train,X_test,y_train,y_test = GetData()
    knn = KNeighborsRegressor(n_neighbors=20)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = knn.score(X_test,y_test)
    plt.scatter(y_test,y_pred)
    plt.xlabel('Scatter For predicted Values')
    
    
    plt.text(80, 10, 'Accuracy = $accuracy$')

    plt.plot
    
    
    
    return accuracy

def DecisionTree():
    X_train,X_test,y_train,y_test = GetData()
    DecisionTree = DecisionTreeRegressor(random_state=0)
    DecisionTree.fit(X_train,y_train)
    y_pred = DecisionTree.predict(X_test)
    
    accuracy = DecisionTree.score(X_test,y_test)
    return accuracy