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


#Getting the data
def GetData():
    X = FM.FeaturesData()
    y = DM.CreateSalesFrame()
    y = y.drop(columns=['Year','Week Number'])
    
    #We create training and testing data that fit with sklearn package

    X_train,X_test,y_train,y_test=ms.train_test_split( X, y, test_size=0.20, random_state=0)


    
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return [X_train,X_test,y_train,y_test]
    #Now We can try our models !