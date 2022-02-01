# Time_Series_Forecasting_SARIMA

#### Prediction Model: Seasonal ARIMA Model
### Libraries Used
1) StatsModels

2) Pandas

3) Seaborn

4) Matplotlib
#### Problem Statement: Temporal forecasting of temperature and Carbon Monoxide (CO) sensor data one day ahead.
### Data Preprocessing
Ran the preprocessing.py on the original dataset to get the current dataset which Is present in the source code. 

Handling -200 values by converting them into NaN values, which were retained to maintain the seasonality of the data while training.

 We are not incorporating the predicted values with NaN values as ground truth for calculating the Mean Absolute Percentage Error(MAPE).
### Model Training and Prediction
For using the Seasonal ARIMA Model for Time Series forecasting the time series must be stationary else we use differencing to make it stationary. The CO(GT) and T columns of the dataset which we had to forecast were already stationary. We are checking it using the Dickey-Fuller test, where the p-value should be less than 0.05 for stationarity.

The seasonal ARIMA model requires p,d,q values for training, where 

1) p is AutoRegressive(AR) Model Lags.

2) d is the number of differencing done to make the series stationary. 

3) q is Moving Average(MA) Lags.

We are plotting Autocorrelation(acf) and Partial Autocorrelation(pacf) to get the p and q parameters, while d is 0 in our case.

We have trained the model in 2 ways. Since it was not clear in the problem description.

1) After predicting for the 8th day, we are discarding the value of the 1st day and adding the ground truth values of the 8th day to the dataset to train the model for predicting the values of the 9th day. We are calling this method "7 days sliding window for training data".

2)  After predicting for the 8th day, we are keeping the value of the 1st day and also adding the ground truth values of the 8th day to the dataset to train the model for predicting the values of the 9th day. We are calling this method "adding data cumulatively to the training set".

We used both methods and forecasted Temperature and Carbon Monoxide values for the next 7 days as mentioned in the procedures.

### Error Analysis (MAPE) For Temperature(T)
#### 7 days sliding window for training data (from 8th to 14th day) :
1) 7.15750108333189

2) 18.41662468287273

3) 10.535269660344241

4) 6.282457707880192

5) 7.600450881824715

6) 24.053844267341738

7) 19.644141084419772

#### Average:  13.38432705257361
#### Adding data cumulatively to the training set (from 8th to 14th day):
1) 7.15750108333189

2) 16.718237478650288

3) 10.641666162168857

4) 7.095582044185502

5) 7.751258948851538

6) 22.42146517791067

7) 19.82212720218842

#### Average: 13.086834013898166
### Error Analysis (MAPE) For Carbon Monoxide(CO)
#### 7 days sliding window for training data (from 8th to 14th day) :
1) 20.431543697088365

2) 18.251331272997575

3) 28.40753180615475

4) 51.26244820005628

5) 35.07352195942088

6) 27.63477342968258

7) 20.30686608381439
#### Average:  28.76685949274497
#### Adding data cumulatively to the training set (from 8th to 14th day):
1) 20.431543697088365

2) 20.527549504914653

3) 27.5187132219326

4) 51.33107825513129

5) 35.455455521341904

6) 37.1597525996471

7) 20.555886056426857
#### Average: 30.425711265211824
### Sample Screenshots

![Alt text](Images/1.png?raw=true "Result1")

![Alt text](Images/2.png?raw=true "Result2")

![Alt text](Images/3.png?raw=true "Result3")

![Alt text](Images/4.png?raw=true "Result4")
