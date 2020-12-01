# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:33:26 2020

@author: maxim
"""

#This file with manage the holiday and exam data
#Holiday and exam dates are gathered in a csv file which for each week indicates if there was a exam or a holiday.

import pandas as pd

def openHandEData():
    return pd.read_csv(r"..\Données\HandEFirstYear.csv", sep=";" ,header = 0)

def distanceToNextExam():
    Table = openHandEData() #Getting the data 
    Exam = [] #This is the column we will add to the data frame containing the distance to the next holiday for every week
    BetweenExamList = []

    for i in Table.index:
        
            indicator = Table['Exam'][i]
            
            #Checking if we aren't in a holiday
            if indicator == 0:
                for l in range(1,len(BetweenExamList)): #The fisrt value correspond to the last holidays so we don't increment it
                    BetweenExamList[l] += 1 #Incrementing the counters
                    
                BetweenExamList.append(1)
            
                
            else :
                #if we are in a holiday then we reset the counter and apply the gotten values to the list
                #Applying the values
                Exam = Exam + BetweenExamList
                                  
                #resetting the list
                BetweenExamList = [0]
                
    
    Exam = Exam + BetweenExamList            
    Table['WeekCounter to Exam'] = Exam
    return Table


#This one probably need to be ponderated depending on the number of exams !
def distanceToNextExam():
    Table = openHandEData() #Getting the data 
    Exams = [] #This is the column we will add to the data frame containing the distance to the next holiday for every week
    BetweenExamList = []

    for i in Table.index:
        
            indicator = Table['Exams'][i]
            
            #Checking if we aren't in a holiday
            if indicator == 0:
                for l in range(1,len(BetweenExamList)): #The fisrt value correspond to the last holidays so we don't increment it
                    BetweenExamList[l] += 1 #Incrementing the counters
                    
                BetweenExamList.append(1)
            
                
            else :
                #if we are in a holiday then we reset the counter and apply the gotten values to the list
                #Applying the values
                Exams = Exams + BetweenExamList
                                  
                #resetting the list
                BetweenExamList = [0]
                
    
    Exams = Exams + BetweenExamList            
    Table['WeekCounter to Exam'] = Exams
    return Table
            

        