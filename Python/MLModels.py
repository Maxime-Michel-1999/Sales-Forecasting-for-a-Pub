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

from sklearn.ensemble import RandomForestRegressor as RFR




#Getting the data
def GetData():
    X = FM.FeaturesData()
    y = DM.CreateSalesFrame()
    
    
    
    for i in y.index :
        
        
        if y['Week Number'][i] < 36 and y['Year'][i] ==  2012 :
            y = y.drop([i])
            
        elif y['Week Number'][i] > 44 and y['Year'][i] == 2019 :
            y = y.drop([i])
            
   
    y = y.drop(columns=['Year','Week Number'])
    
    
    
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
    knn = KNeighborsRegressor(n_neighbors=15,n_jobs=6)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)

    print(len(y_pred))
    accuracy = round( knn.score(X_test,y_test),3)
    plt.scatter(y_test,y_pred)
    plt.title('Scatter For predicted Values')
    text = "Accuracy = {} ".format(accuracy)
    plt.text(150, 10,text,fontsize='x-large')
    plt.show()
    
    
    
    return accuracy

def DecisionTree():
    X_train,X_test,y_train,y_test = GetData()
    DecisionTree = DecisionTreeRegressor(random_state=0)
    DecisionTree.fit(X_train,y_train)
    y_pred = DecisionTree.predict(X_test)
    
    accuracy = round(DecisionTree.score(X_test,y_test),3)
    

    plt.scatter(y_test,y_pred)
    plt.title('Scatter For predicted Values')
    text = "Accuracy = {} ".format(accuracy)
    plt.text(80, 10,text,fontsize='x-large')
    plt.show()
    
    
def RandomForestRegressor():
    X_train,X_test,y_train,y_test = GetData()
    RandomForest = RFR(n_estimators=400, max_depth=15, n_jobs=6)
    RandomForest.fit(X_train,y_train)
    y_pred = RandomForest.predict(X_test)
    
    accuracy = round(RandomForest.score(X_test,y_test),3)
    

    plt.scatter(y_test,y_pred)
    plt.title('Scatter For predicted Values')
    text = "Accuracy = {} ".format(accuracy)
    plt.text(80, 10,text,fontsize='x-large')
    plt.show()
    
