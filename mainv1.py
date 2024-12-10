import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read and save data from csv file, do not forget to change datafile_name
data_file = "train.csv"
data = pd.read_csv(data_file)
data.head() # shows top 5 rows of data, use this to check that the data loaded correctly
print(data.head())

#shows missing data
missing_data = data.isna()
Amount_Missing_Data = data.isna().sum().sum()
print(Amount_Missing_Data)

#checks for duplication and the amount of duplication:
duplication = data.duplicated()
duplicated_rows = []
for i in range(len(duplication)): 
    if duplication[i] == True:
        duplicated_rows.append(duplication[i])

print(len(duplicated_rows))
if len(duplicated_rows) > 0:
    print(duplicated_rows)

