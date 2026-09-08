import pandas as pd
import matplotlib.pyplot as plt

with open("49_Industry_Portfolios.csv") as file:
    lines = file.readlines()
    
HeaderSpace = 9
FirstBlank = 0

for i, line in enumerate(lines[HeaderSpace+1:], start = HeaderSpace + 1):
    if line.strip() == "":
        FirstBlank = i
        break

nrows = FirstBlank - (HeaderSpace+1)

StockReturns = pd.read_csv(
    "49_Industry_Portfolios.csv",
    skiprows=HeaderSpace,
    nrows=nrows,
    index_col = 0,
    na_values = [-99.99,-999],
)

print(StockReturns.head())