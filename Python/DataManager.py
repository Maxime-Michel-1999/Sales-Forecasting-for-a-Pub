#This file is made to open and gather the data in python friendly object

import pandas as pd
#The sales array will gather all the sales for each category for each week
Sales = []


#Creating Table with each week for the 10 last years

def ProductSales(df):
    
    weekIndex = 0
    salesCounter = []
    salesCounterIndex = -1
    for i in df.index:
        date = df["timestamp"][i]
        weekNo = date.isocalendar()[1]
        if weekNo == weekIndex :
            salesCounter[salesCounterIndex][2] += 1
        
        else :
            salesCounterIndex += 1
            weekIndex = weekNo
            salesCounter.append([date.year,weekIndex,1])
        
        #salesCounter is now an object containing the year, the week number and the number of sales.
        return salesCounter

def CreateDataFrame():
    
    
    #Normal Beers Object
    NormalBeer = pd.read_csv(r"..\Données\NormalBeer.csv", sep=";" ,header = 0, parse_dates = ["timestamp"])
    NormalBeerSales = ProductSales(NormalBeer)

    
    #HighBeer
    HighBeer = pd.read_csv(r"..\Données\HighBeer.csv", sep=";" ,header = 0, parse_dates = ["timestamp"])
    HighBeerSales = ProductSales(NHighBeer)

    #NotBeer
    NotBeer = pd.read_csv(r"..\Données\HighBeer.csv", sep=";" ,header = 0, parse_dates = ["timestamp"])
    NotBeerSales = ProductSales(NotBeer)
    
    return [NormalBeerSales, HighBeerSales, NotBeerSales]


def CreateSalesData(SalesTables):
    #The goal here is to create a data frame for every year with for each week the sales of each categories
    
    #Checking the number of number of year
    years = []
    for i in SalesTable[0]:
        year = i[0]
        if year not in years:
            years.append(year)
    print('yo mona')
    
    
    #Now we create a data frame for every year 
    for j in years :
        
        
        
    
    
    
def TestTrainSeparator():
    