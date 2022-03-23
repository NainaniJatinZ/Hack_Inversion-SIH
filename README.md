# Hack_Inversion-SIH

We propose a Machine learning price-forecasting model that takes a more balanced overview at the significant price affecting factors viz. technical indicators (Eg: MACD, EMA, a custom indicator) and real-world factors such as NG production, storage, logistics, and weather data. 

We also perform trend analysis on the data by locating the instances of technical patterns. 

We have used three types of models (LSTM, Statistical, and SVR) to compare the best forecast. Forecasting is done recursively, feeding predictions to forecast future data. 
The forecasts are compared with benchmarks through RMSE, M-DM(Multivariate Diebold Mariano test), and AIC(Akaike information criterion). 

A Website is also deployed to provide a dashboard for visualizations and forecasted results as output. 

In this machine learning model, we are using three independent models, viz., LSTM, Hybrid Statistical model and SVR. 

The LSTM model leverages various technical indicators, namely, MACD, 21w EMA, 20w SMA, Custom Index, closing value and volume, in order to accurately forecast Natural Gas prices. We are using CNN LSTM and Encoder-Decoder LSTM model, and the best of these two will be selected. 

For hybrid statistical model we are focussing only on closing prices, and for the SVR model 3 day SMA as input to reduce the variance and help capture the non linearity in the price data.


## Algorithm: 

![sih-model (1)](https://user-images.githubusercontent.com/68293556/159704425-127ae5e4-012a-48d5-933c-921755399d65.png)


![sih2 (1)](https://user-images.githubusercontent.com/68293556/159704371-c6b1f9af-efb6-4c83-9640-ddda858aa27e.png)


## Indicators:

1. Custom Index:
![Custom_index_v2 (1)](https://user-images.githubusercontent.com/68293556/159704652-ca1ed414-7c66-4ae7-95ba-81c0ae1834f2.PNG)

2. EMA and SMA
![ema_sma_nymex (1)](https://user-images.githubusercontent.com/68293556/159704698-8a34e5e6-8898-4465-b26c-b861b8bbaa34.png)

## Pattern Recognition:

![pattern recog](https://user-images.githubusercontent.com/68293556/159704803-4483cf8a-72b5-4ee3-8fbe-e8d19df1c1f2.png)



