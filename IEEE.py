
## Importing Libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

## Data Preprocessing

data = "./AirQuality.csv"
df = pd.read_csv(data)
df["DateTime"] = df["Date"] + " " + df["Time"]
df = df[["DateTime", "CO(GT)", "T"]]
df = df.replace(-200, np.nan)
df = df[6:]
df[:168]

## Forecasting Temparature(T)

sns.set(rc = {"figure.figsize" : (96, 48)})
sns.lineplot(x = "DateTime", y = "T", data = df[:168].dropna())

# Dickey Fuller Test 

from statsmodels.tsa.stattools import adfuller
X = df["T"][:168].dropna()
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# p-value < 0.005 for stattionary

# Autocorrelation and Partial Autocorrelation
  
fig = plt.figure(figsize = (12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df["T"].iloc[:168].dropna(), lags = 50, ax = ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df["T"].iloc[:168].dropna(), lags = 50, ax = ax2)

# p = 1 Autregrssion model lags
# d = 0 Differencing
# q = 0 Moving Average lags

# Prediction Examples for 
# 1) 7 day sliding window
# 2) Adding data cumulatively to the training set  

model = sm.tsa.statespace.SARIMAX(df["T"].iloc[96:264], order = (1, 0, 0), seasonal_order = (1, 0, 0, 24))
results = model.fit()
df['T_forecast'] = results.predict(start = 168, end = 192, dynamic = True)
df[['T','T_forecast']].iloc[200:300].plot(figsize=(12,8))

model = sm.tsa.statespace.SARIMAX(df["T"].iloc[:168], order = (1, 0, 0), seasonal_order = (1, 0, 0, 24))
results = model.fit()
df['T_forecast'] = results.predict(start = 168, end = 192, dynamic = True)
df[['T','T_forecast']].iloc[100:200].plot(figsize=(12,8))

# Error Analysis for 7 days sliding window for training data 

l = []
for i in range(0, 7):
  model = sm.tsa.statespace.SARIMAX(df["T"].iloc[i * 24: i * 24 + 168], order = (1, 0, 0), seasonal_order = (1, 0, 0, 24))
  results = model.fit()
  df['T_forecast'] = results.predict(start = 168, end = 192, dynamic = True)
  n = df["T"][i * 24 + 168 : i * 24 + 192].isna().sum()
  one = ((abs(((df["T"][i * 24 + 168 : i * 24 + 192] - df["T_forecast"][i * 24 + 168 : i * 24 + 192]) / df["T"][i * 24 + 168 : i * 24 + 192]).dropna()).sum()) / (24 - n)) * 100
  print(one)
  l.append(one)

# 7 day sliding window

print(sum(l) / 7)

# Error Analysis for Adding data cumulatively to the training set

l = []
for i in range(0, 7):
  model = sm.tsa.statespace.SARIMAX(df["T"].iloc[: i * 24 + 168], order = (1, 0, 0), seasonal_order = (1, 0, 0, 24))
  results = model.fit()
  df['T_forecast'] = results.predict(start = 168 + i * 24, end = 192 + i * 24, dynamic = True)
  n = df["T"][i * 24 + 168 : i * 24 + 192].isna().sum()
  one = ((abs(((df["T"][i * 24 + 168 : i * 24 + 192] - df["T_forecast"][i * 24 + 168 : i * 24 + 192]) / df["T"][i * 24 + 168 : i * 24 + 192]).dropna()).sum()) / (24 - n)) * 100
  print(one)
  l.append(one)

# Cumulative

print(sum(l) / 7)

## Forecasting Carbon Monoxide(CO)

sns.set(rc = {"figure.figsize" : (96, 48)})
sns.lineplot(x = "DateTime", y = "CO(GT)", data = df[:168].dropna())

# Dickey Fuller Test 

from statsmodels.tsa.stattools import adfuller
X = df["CO(GT)"][:168].dropna()
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#p-value < 0.005 for stattionary apna 0.0000 hai

# Autocorrelation and Partial Autocorrelation

fig = plt.figure(figsize = (12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df["CO(GT)"].iloc[:168].dropna(), lags = 50, ax = ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df["CO(GT)"].iloc[:168].dropna(), lags = 50, ax = ax2)

# p = 1 Autregrssion model lags
# d = 0 Differencing
# q = 1 for T Moving Average lags

# Prediction Examples for 
# 1) 7 day sliding window
# 2) Adding data cumulatively to the training set

model = sm.tsa.statespace.SARIMAX(df["CO(GT)"].iloc[96: 264], order = (1, 0, 1), seasonal_order = (1, 0, 1, 24))
results = model.fit()
df['CO_forecast'] = results.predict(start = 168, end = 192, dynamic = True)
df[['CO(GT)','CO_forecast']].iloc[200:300].plot(figsize=(12,8))

model = sm.tsa.statespace.SARIMAX(df["CO(GT)"].iloc[:168], order = (1, 0, 1), seasonal_order = (1, 0, 1, 24))
results = model.fit()
df['CO_forecast'] = results.predict(start = 168, end = 192, dynamic = True)
df[['CO(GT)','CO_forecast']].iloc[100:250].plot(figsize=(12,8))

# Error Analaysis for 7 days sliding window for training data 

l = []
for i in range(0, 7):
  model = sm.tsa.statespace.SARIMAX(df["CO(GT)"].iloc[i * 24: i * 24 + 168], order = (1, 0, 1), seasonal_order = (1, 0, 1, 24))
  results = model.fit()
  df['CO_forecast'] = results.predict(start = 168, end = 192, dynamic = True)
  n = df["CO(GT)"][i * 24 + 168 : i * 24 + 192].isna().sum()
  one = ((abs(((df["CO(GT)"][i * 24 + 168 : i * 24 + 192] - df["CO_forecast"][i * 24 + 168 : i * 24 + 192]) / df["CO(GT)"][i * 24 + 168 : i * 24 + 192]).dropna()).sum()) / (24 - n)) * 100
  print(one)
  l.append(one)

# 7 day sliding window

print(sum(l) / 7)

# Error Analysis for Adding data cumulatively to the training set

l = []
for i in range(0, 7):
  model = sm.tsa.statespace.SARIMAX(df["CO(GT)"].iloc[: i * 24 + 168], order = (1, 0, 1), seasonal_order = (1, 0, 1, 24))
  results = model.fit()
  df['CO_forecast'] = results.predict(start = 168 + i * 24, end = 192 + i * 24, dynamic = True)
  n = df["CO(GT)"][i * 24 + 168 : i * 24 + 192].isna().sum()
  one = ((abs(((df["CO(GT)"][i * 24 + 168 : i * 24 + 192] - df["CO_forecast"][i * 24 + 168 : i * 24 + 192]) / df["CO(GT)"][i * 24 + 168 : i * 24 + 192]).dropna()).sum()) / (24 - n)) * 100
  print(one)
  l.append(one)

# Cumulative

print(sum(l) / 7)

