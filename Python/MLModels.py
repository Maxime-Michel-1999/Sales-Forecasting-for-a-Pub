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
from sklearn.metrics import mean_squared_error as meanSquared
from sklearn.metrics import mean_absolute_error as meanAbs


from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import ExtraTreesRegressor

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
    
   
    
    X = X.drop(columns=['Week to Holiday'])

    
    #We create training and testing data that fit with sklearn package

    X_train,X_test,y_train,y_test=ms.train_test_split( X, y, test_size=0.20, random_state = 0,shuffle=False)

    #Do we need this ?
    
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    

    return  X_train,X_test,y_train,y_test

    #Now We can try our models !

def KNN_Regressor(X_train,X_test,y_train,y_test):
    
    knn = KNeighborsRegressor(n_neighbors=15,n_jobs=6)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    
    accuracy,crossScore,meanSquare,meanAbsolute = Scores(knn,X_test,y_test,y_pred)
    
    DisplayResults(y_test,y_pred,accuracy,crossScore,meanSquare,meanAbsolute)



def DecisionTree(X_train,X_test,y_train,y_test):

    DecisionTree = DecisionTreeRegressor(random_state=0)
    DecisionTree.fit(X_train,y_train)
    y_pred = DecisionTree.predict(X_test) 
    
    accuracy,crossScore,meanSquare,meanAbsolute = Scores(DecisionTree,X_test,y_test,y_pred)
    
    DisplayResults(y_test,y_pred,accuracy,crossScore,meanSquare,meanAbsolute)


      
def RandomForestRegressor(X_train,X_test,y_train,y_test):

    RandomForest = RFR(n_estimators=100)
    RandomForest.fit(X_train,y_train)
    y_pred = RandomForest.predict(X_test)
    
    accuracy,crossScore,meanSquare,meanAbsolute = Scores(RandomForest,X_test,y_test,y_pred)
    DisplayResults(y_test,y_pred,accuracy,crossScore,meanSquare,meanAbsolute)

    return y_pred

def NeuralNet(X_train,X_test,y_train,y_test):
 
    Neural = MLPRegressor(random_state=1, max_iter=3000)
    Neural.fit(X_train, y_train)
    y_pred = Neural.predict(X_test)
    
    accuracy,crossScore,meanSquare,meanAbsolute = Scores(Neural,X_test,y_test,y_pred)
    
    DisplayResults(y_test,y_pred,accuracy,crossScore,meanSquare,meanAbsolute)
    


def ExtraTreeRegressor(X_train,X_test,y_train,y_test):
    etr = ExtraTreesRegressor(n_estimators=30,n_jobs=4) 
    etr.fit(X_train,y_train)
    y_pred=etr.predict(X_test)
    
    accuracy,crossScore,meanSquare,meanAbsolute = Scores(etr,X_test,y_test,y_pred)
    
    DisplayResults(y_test,y_pred,accuracy,crossScore,meanSquare,meanAbsolute)



    
def Scores(model,X_test,y_test,y_pred):
    
    accuracy = round(model.score(X_test,y_test),3)
    crossScore = CV(model,X_test,y_test,scoring="neg_mean_absolute_error")
    meanSquare = meanSquared(y_test,y_pred)
    meanAbsolute = meanAbs(y_test,y_pred)
    return accuracy,crossScore,meanSquare,meanAbsolute
    
    
def DisplayResults(y_test,y_pred,accuracy,crossScore,meanSquare,meanAbsolute):
    
    
    
    
    
    plt.scatter(y_test,y_pred)
    plt.title('Predicted Values')
    plt.xlabel("Real Values")
    plt.ylabel("Predicted Values")
    plt.show()
    
    #print("CrossValidation scores : " , crossScore)
    print("Accuracy : " , round(accuracy,2))
    print("MSE : " , round(meanSquare,2))
    print("MAE : ",round(meanAbsolute,2))
    
    
def TestModels():
    X_train,X_test,y_train,y_test = GetData()
    
    print("KNN Regressor :")
    KNN_Regressor(X_train,X_test,y_train,y_test)
    
    print("\n Decision Tree :")
    DecisionTree(X_train,X_test,y_train,y_test)
    
    print("\n Random Forest :")
    RandomForestRegressor(X_train,X_test,y_train,y_test)
  
    
    print("\n Neural Network :")
    NeuralNet(X_train,X_test,y_train,y_test)

    print("\n Extra Tree Regressor :")
    ExtraTreeRegressor(X_train,X_test,y_train,y_test)

def concreteResults():
    X_train,X_test,y_train,y_test = GetData()
    y_pred = RandomForestRegressor(X_train,X_test,y_train,y_test)
    j=0
    for i in y_test.columns:
        
        Normal = y_test[i].values.tolist()
        NormalPred = y_pred[:,j]
        j += 1
    
        plt.plot(NormalPred, label = 'Prediction', color = "blue")
        plt.plot(Normal, label = 'Real', color = "green")
        plt.title(i)
        plt.show()
    
    
    
    
    
    
    
#Let's try some basic methods usually used : 
    