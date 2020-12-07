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

def distanceToNextHoliday():
    Table = openHandEData() #Getting the data 
    Holiday = [] #This is the column we will add to the data frame containing the distance to the next holiday for every week
    BetweenHolidayList = []

    for i in Table.index:
        
            indicator = Table['Holiday'][i]
            
            #Checking if we aren't in a holiday
            if indicator == 0:
                for l in range(1,len(BetweenHolidayList)): #The fisrt value correspond to the last holidays so we don't increment it
                    BetweenHolidayList[l] += 1 #Incrementing the counters
                    
                BetweenHolidayList.append(1)
            
                
            else :
                #if we are in a holiday then we reset the counter and apply the gotten values to the list
                #Applying the values
                Holiday = Holiday + BetweenHolidayList
                                  
                #resetting the list
                BetweenHolidayList = [0]
                
    
    Holiday = Holiday + BetweenHolidayList            
    return Holiday


#This one probably need to be ponderated depending on the number of exams !
def distanceToNextExam():
    Table = openHandEData() #Getting the data 
    Exams = [] #This is the column we will add to the data frame containing the distance to the next holiday for every week

    for i in Table.index:
            coefficient = 0
            ponderate = 3
            for j in range(i+1,i+4): #the goal is two analyse the three following weeks
                try :
                    coefficient += (Table['Exams'][j])*ponderate
                    
                except:
                    coefficient += 0
                        
                ponderate -= 1
                
            #We now have a coefficient for each week indicating the coming exams !

            
            Exams.append(coefficient)    
            
    return Exams
            
def CreateHandE():
    
    Holiday = distanceToNextHoliday()
    Exam = distanceToNextExam()
    print(len(Holiday))
    print(len(Exam))
    
    data = {'Week to Holiday': Holiday,
        'Exam Coefficient': Exam
        }

    
    dataFrame = pd.DataFrame(data,columns=['Week to Holiday', 'Exam Coefficient'])
    
    return dataFrame
        

def FeaturesData():
    HandE = CreateHandE()
    
    #we had the week number and the promotion (which are features).
    
    CSV  = openHandEData()
    week = []
    promotion =[]
    for i in CSV.index:
        week.append(CSV['WeekNumber'][i])
        promotion.append(CSV['Promotion'][i])
    
    HandE['WeekNumber'] = week
    HandE['Promotion'] = promotion
    return HandE
    
    
        
        
    
    
    
    
    
    
    
    
    
    