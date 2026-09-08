import pandas as pd
import matplotlib.pyplot as plt

with open("49_Industry_Portfolios.csv") as file:
    lines = file.readlines()
    
HeaderSpace = 11
FirstBlank = 0

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

#Change the date to the right format.
StockReturns.index = pd.to_datetime(StockReturns.index, format="%Y%m")
StockReturns.index.name = "Date"
StockReturns = StockReturns.sort_index()

print(StockReturns["Agric"].head())