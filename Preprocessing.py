import pandas as pd

df = pd.read_csv(".\AirQuality.csv", delimiter = ';')
tempdf = df[:][:9357]
tempdf = tempdf[['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)','PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)','PT08.S5(O3)', 'T', 'RH', 'AH']]
columns = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']

for column in columns:  
    for row in tempdf.index:
        tempdf[column][row] = tempdf[column][row].replace(",", ".")
print(tempdf)            
tempdf.to_csv("./airquality.csv", index = False)