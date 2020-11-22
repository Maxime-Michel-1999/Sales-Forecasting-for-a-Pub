#This file is made to open and gather the data in python friendly object

import pandas as pd
#The sales array will gather all the sales for each category for each week


#Creating Table with each week for the 10 last years


def ManageNullWeek(Sales):
    #This function will manage the issue of having null row, by just adding a row with 0 sales if there aren't any
    indexWeek = 1
    NewSales = []
    index = 0
    while index < len(Sales):
        
        if indexWeek>53:
            indexWeek=1
        
        if Sales[index][1] == indexWeek:
            NewSales.append(Sales[index])
            index = index + 1
            indexWeek = indexWeek  + 1
            
        else :
            NewSales.append([Sales[index][0],indexWeek,0])
            indexWeek = indexWeek + 1
    

    return NewSales

def ProductSales(df):
    

    weekIndex = 1
    salesCounter = []
    salesCounterIndex = -1
    for i in df.index:
        date = df["timestamp"][i]
        weekNo = date.isocalendar()[1]
        if weekNo == weekIndex :
            if salesCounterIndex < 0 :
                salesCounterIndex = 0
                salesCounter.append([date.year,weekIndex,1])
            salesCounter[salesCounterIndex][2] += 1
        
        else :
            salesCounterIndex += 1
            weekIndex = weekNo
            salesCounter.append([date.year,weekIndex,1])
        
        #salesCounter is now an object containing the year, the week number and the number of sales.
        #Complete Null rows
    salesCounter = ManageNullWeek(salesCounter)
    return salesCounter
    
    

def CreateSalesArray():
    
    
    #Normal Beers Object
    NormalBeer = pd.read_csv(r"..\Données\NormalBeer.csv", sep=";" ,header = 0, parse_dates = ["timestamp"])
    NormalBeerSales = ProductSales(NormalBeer)

    
    #HighBeer
    HighBeer = pd.read_csv(r"..\Données\HighBeer.csv", sep=";" ,header = 0, parse_dates = ["timestamp"])
    HighBeerSales = ProductSales(HighBeer)

    #NotBeer
    NotBeer = pd.read_csv(r"..\Données\NotBeer.csv", sep=";" ,header = 0, parse_dates = ["timestamp"])
    NotBeerSales = ProductSales(NotBeer)
    
    #Special Beer
    SpecialBeer = pd.read_csv(r"..\Données\SpecialBeer.csv", sep=";" ,header = 0, parse_dates = ["timestamp"])
    SpecialBeerSales = ProductSales(SpecialBeer)
    
    return [NormalBeerSales, HighBeerSales, NotBeerSales, SpecialBeerSales]

            
        

def CreateSalesData(SalesTable):
    #The goal here is to create a data frame for every year with for each week the sales of each categories
    
    #Checking the number of number of year
    
    years = []
    
      
    SalesbyYear = dict() #this will contain the dataframe for each year
        
    for i in SalesTable[0]:
        year = i[0]
        if year not in years:
            years.append(year)
    
    
    yearIndex = 0
    #Now we create a data frame for every year 
    for j in years :
        
        WeekSales = []
        for l in range(52):
            #Adding a row with each sales for each categories (3 for the moment)
            # !!!! Don't forget to change and add a column for added categories !!!
            try :
                WeekSales.append([l +1,SalesTable[0][l + yearIndex*52][2],SalesTable[1][l + yearIndex*52][2],SalesTable[2][l + yearIndex*52][2],SalesTable[3][l + yearIndex*52][2]])
            except :
                break
        
        SalesbyYear[j] = ( pd.DataFrame(WeekSales,columns=['Week Number','Normal Beer', 'High Degree Beer', 'Not Beer', 'Special Beer']))
        yearIndex += 1
    return(SalesbyYear)
            
        
        
#This Function is the main one, it gather all the work and gives the sales dataframe !
def CreateSalesFrame():      
    SalesTable = CreateSalesArray()
    table = CreateSalesData(SalesTable)
    return table
    
    