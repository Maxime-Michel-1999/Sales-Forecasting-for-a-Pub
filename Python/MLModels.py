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
from sklearn.model_selection import cross_val_score as CV

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings("ignore")



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

    #Now We can try our models !

def KNN_Regressor(X_train,X_test,y_train,y_test):
    
    knn = KNeighborsRegressor(n_neighbors=15,n_jobs=6)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = round(knn.score(X_test,y_test),3)
    crossScore = CV(knn,X_test,y_test)
    DisplayResults(y_test,y_pred,accuracy,crossScore)


def DecisionTree(X_train,X_test,y_train,y_test):

    DecisionTree = DecisionTreeRegressor(random_state=0)
    DecisionTree.fit(X_train,y_train)
    y_pred = DecisionTree.predict(X_test) 
    
    accuracy = round(DecisionTree.score(X_test,y_test),3)
    crossScore = CV(DecisionTree,X_test,y_test)
    DisplayResults(y_test,y_pred,accuracy,crossScore)

      
def RandomForestRegressor(X_train,X_test,y_train,y_test):

    RandomForest = RFR(n_estimators=400, max_depth=15, n_jobs=6)
    RandomForest.fit(X_train,y_train)
    y_pred = RandomForest.predict(X_test)
    
    accuracy = round(RandomForest.score(X_test,y_test),3)  
    crossScore = CV(RandomForest,X_test,y_test)
    DisplayResults(y_test,y_pred,accuracy,crossScore)


def NeuralNet(X_train,X_test,y_train,y_test):
 
    Neural = MLPRegressor(random_state=1, max_iter=5000)
    Neural.fit(X_train, y_train)
    y_pred = Neural.predict(X_test)
    
    accuracy = round(Neural.score(X_test,y_test),3)
    crossScore = CV(Neural,X_test,y_test)
    DisplayResults(y_test,y_pred,accuracy,crossScore)

    
def DisplayResults(y_test,y_pred,accuracy,crossScore):
    plt.scatter(y_test,y_pred)
    plt.title('Scatter For predicted Values')
    text = "Accuracy = {} ".format(accuracy)
    plt.text(200, 10,text,fontsize='x-large')
    plt.show()
    print("CrossValidation scores : " , crossScore)
    
    
def TestModels():
    X_train,X_test,y_train,y_test = GetData()
    
    print("KNN Regressor :")
    KNN_Regressor(X_train,X_test,y_train,y_test)
    
    print("Decision Tree :")
    DecisionTree(X_train,X_test,y_train,y_test)
    
    print("Random Forest :")
    RandomForestRegressor(X_train,X_test,y_train,y_test)
    
    print("Neural Network :")
    NeuralNet(X_train,X_test,y_train,y_test)


    
    