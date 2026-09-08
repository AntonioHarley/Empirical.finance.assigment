import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open("49_Industry_Portfolios.csv") as file:
    lines = file.readlines()
    
#The industries we chose for the assignment.
Industries = ["Oil", "Banks", "Txtls", "Toys", "Rtail"]

#Size header of the csv. 
HeaderSpace = 11
FirstBlank = 0

#We search for the line that is blank after the header of the csv.
for i, line in enumerate(lines[HeaderSpace+1:], start = HeaderSpace + 1):
    if line.strip() == "":
        FirstBlank = i
        break

nrows = FirstBlank - (HeaderSpace+1)

#Read the csv-file for the 49 industries with the skip at the beginning.
StockReturns = pd.read_csv(
    "49_Industry_Portfolios.csv",
    skiprows=HeaderSpace,
    nrows=nrows,
    index_col=0,
    na_values=[-99.99, -999],
)

#Strip padding spaces from the industry column names.
StockReturns.columns = StockReturns.columns.str.strip()

#Change the date to the right format.
StockReturns.index = pd.to_datetime(StockReturns.index, format="%Y%m")
StockReturns.index.name = "Date"
StockReturns = StockReturns.sort_index()

#Check if everything went well.
print(StockReturns["Agric"].head())


for industry in Industries:
    #Basic stats check
    mean = np.mean(StockReturns[industry])
    std = np.std(StockReturns[industry]) 
    print( f"{mean} mean {industry} returns")
    print(f"{std} standard  deviation {industry} returns")
    
    
    #Histogram for the monthly returns.
    plt.hist(StockReturns[industry], bins = 75, density = False)
    plt.xlabel(f"{industry} monthly return (%)")
    plt.ylabel("Frequency")
    plt.title(f"{industry} — Average Value Weighted Montly Returns")
    plt.show()
    
    #Time-series for the montly returns.
    plt.plot(StockReturns[industry])
    plt.title(industry)
    plt.xlabel("Date")
    plt.ylabel("Monthly return (%)")
    plt.show()

