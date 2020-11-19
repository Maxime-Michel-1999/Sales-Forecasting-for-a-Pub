#This file is made to open and gather the data in python friendly object

import pandas as pd
#The sales array will gather all the sales for each category for each week
Sales = []


#Creating Table with each week for the 10 last years





#Normal Beers Object
NormalBeer = pd.read_csv(r"..\Données\NormalBeer.csv", sep=";" ,header = 0, parse_dates = ["timestamp"])

weekIndex = 0
salesCounter = []
salesCounterIndex = -1
counter = 0
for i in NormalBeer.index:
    date = NormalBeer["timestamp"][i]
    weekNo = date.isocalendar()[1]
    if weekNo == weekIndex :
        salesCounter[salesCounterIndex][2] += 1
        
    else :
        salesCounterIndex += 1
        weekIndex = weekNo
        salesCounter.append([date.year,weekIndex,1])
        
    counter += 1

#salesCounter is now an object containing the year, the week number and the number of sales.

#print(NormalBeer.loc[1][1])





date = NormalBeer.loc[1][1]
print(date)
print(date.isocalendar()[1])