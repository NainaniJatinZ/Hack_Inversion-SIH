import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
from sklearn.metrics import mean_absolute_error
import pickle



def prep(data):
    data = data[['Date', 'Close', 'Volume']]
    data = data.fillna(data.mean())
    data.isna().sum()
    return data

def calculate_sma(data_series, window_size):
  windows = data_series.rolling(window_size)
  moving_averages = windows.mean()
  moving_averages_list = moving_averages.tolist()
  without_nans = moving_averages_list[window_size - 1:]
  # return without_nans
  moving_averages_list = [0 if math.isnan(x) else x for x in moving_averages_list]
  return moving_averages_list

def calculate_ema(prices, window_size, smoothing=2):
    ema = [sum(prices[:window_size]) / window_size]
    for price in prices[window_size:]:
        ema.append((price * (smoothing / (1 + window_size))) + ema[-1] * (1 - (smoothing / (1 + window_size))))
    # print(len(ema))
    
    emaList = [0 for i in range(window_size-1)]
    emaList = [y for x in [emaList, ema] for y in x]
    # emaList = emaList + ema
    return emaList

def create_indicators(data):
    ma = calculate_sma(data['Close'], 21*7)
    ema = calculate_ema(data['Close'], 21*7)

    ema1 = calculate_ema(data['Close'], 12*7)
    ema2 = calculate_ema(data['Close'], 26*7)

    macd = list()
    for i in range(len(ema1)):
        macd.append(ema1[i] - ema2[i])

    u = [sum(x)/2 for x in zip(ma, ema)]
    Kr = 0.3
    Kp = 0.9
    custom = list()
    for i in range(len(u)):
        P = data['Close'][i]
        if (P > u[i]*(1 - Kr)) and (P < u[i]*(1 + Kr)):

            x = ((P - u[i])*Kp*100)/(u[i]*Kr)

        elif (P < u[i]*(1 - Kr)):

            x = (P*(1 - Kp)/(u[i]*(1 - Kr)) - 1)*100

        else:
            
            x = (1 - (u[i]*(1 + Kr)*(1 - Kp))/P)*100
        custom.append(x)

    data['SMA']= pd.Series(ma)
    data['EMA']= pd.Series(ema)
    data['MACD']= pd.Series(macd)
    data['Custom'] = pd.Series(custom)

    return data

def train_test(data):        
    data = data[181:]

    data = data.reset_index()
    data = data.drop(columns = ['index'])

    train_df = data.iloc[:int(len(data)*0.7)]
    test_df = data.iloc[int(len(data)*0.7):]

    train = train_df
    scaler = MinMaxScaler(feature_range=(-4,4))
    s_s = scaler.fit_transform(train['Volume'].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    train['Volume']=s_s
    test = test_df
    s_s = scaler.transform(test['Volume'].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    test['Volume']=s_s

    train = train.drop(columns = ['Date'])
    test = test.drop(columns = ['Date'])

    return train, test, data

def lstm_mae(train, test, data,n_future=1000):
    lstm = tf.keras.models.load_model("models/model_e4d4.h5")

    # n_future = 1000

    init_test = np.expand_dims(test.iloc[:50, :], axis=0)
    init_test = np.asarray(init_test).astype(np.float32)

    for i in range(int(n_future/30)):
        pred_init = lstm.predict(init_test[:,-50:,:])
        init_test = np.concatenate((init_test, pred_init), axis = 1)

    pred_val = init_test[:, :, 0]
    test_index = []
    for i in range(int(len(data)*0.7), int(len(data)*0.7)+ init_test.shape[1]):
        test_index.append(i)
    # print(int(len(data)*0.7))
    fc1 = pd.Series(pred_val[0], index=test_index)
    return pred_val[0] , test_index

def lstm(train, test, data,n_future = 1000):
    lstm = tf.keras.models.load_model("models/model_e4d4.h5")

    

    init_test = np.expand_dims(test.iloc[-50:, :], axis=0)
    init_test = np.asarray(init_test).astype(np.float32)

    for i in range(int(n_future/30)):
        pred_init = lstm.predict(init_test[:,-50:,:])
        init_test = np.concatenate((init_test, pred_init), axis = 1)

    pred_val = init_test[:, :, 0]
    test_index = []
    for i in range(int(len(data)), int(len(data))+ init_test.shape[1]):
        test_index.append(i)
    # print(int(len(data)*0.7))
    # fc1 = pd.Series(pred_val[0], index=test_index)
    prev = data["Close"].values.tolist()
    return pred_val[0] , test_index, prev

def arima_es_mae(data,n_future = 1000):
    
    arima = joblib.load('models/arima_direct_fore.pkl')
    es = joblib.load('models/expo_hope.pkl')
    fore, se, conf = arima.forecast(n_future+50, alpha=0.05)
    print(fore.shape)

    test_index1 = []
    for i in range(int(len(data)*0.7), int(len(data)*0.7)+ fore.shape[0]):
        test_index1.append(i)



    fc_series = pd.Series(fore, index=test_index1)
    lower_series = pd.Series(conf[:, 0], index=test_index1)
    upper_series = pd.Series(conf[:, 1], index=test_index1)

    predictions_mul = es.forecast(steps=n_future+50)

    fc_series1 = pd.Series(predictions_mul.values.tolist(), index=test_index1)
    lower_series = pd.Series(conf[:, 0], index=test_index1)
    upper_series = pd.Series(conf[:, 1], index=test_index1)

    w1 = 1 
    w2 = 0.1

    fore_this_cast = fc_series*w1  + fc_series1*w2

    vals=fore_this_cast.values.tolist()
    return vals, test_index1


def arima_es(train, test, data,n_future = 1000):
    
    arima = joblib.load('models/arima_direct_fore.pkl')
    es = joblib.load('models/expo_hope.pkl')

    fore, se, conf = arima.forecast(n_future+50, alpha=0.05)
    # print(fore.shape)

    test_index1 = []
    for i in range(int(len(data)), int(len(data))+ fore.shape[0]):
        test_index1.append(i)



    fc_series = pd.Series(fore, index=test_index1)
    lower_series = pd.Series(conf[:, 0], index=test_index1)
    upper_series = pd.Series(conf[:, 1], index=test_index1)

    predictions_mul = es.forecast(steps=n_future+50)

    fc_series1 = pd.Series(predictions_mul.values.tolist(), index=test_index1)
    lower_series = pd.Series(conf[:, 0], index=test_index1)
    upper_series = pd.Series(conf[:, 1], index=test_index1)

    w1 = 1 
    w2 = 0.1

    fore_this_cast = fc_series*w1  + fc_series1*w2

    vals=fore_this_cast.values.tolist()
    prev = data["Close"].values.tolist()
    return vals, test_index1, prev

def svr_mae(train, test, data,n_future = 1000):
    
    with open('models/svr_final.pkl', 'rb') as pickle_file:
        svm = pickle.load(pickle_file)

    init = np.expand_dims(test['Close'][:30], axis=0)

    for i in range(n_future):
        pred=np.expand_dims(svm.predict(init[:, -30:]), axis=0)
        init = np.concatenate((init, pred), axis=1)
    print(init.shape)
    test_index_svr = []
    for i in range(int(len(data)*0.7), int(len(data)*0.7)+ init.shape[1]):
        test_index_svr.append(i)

    forget_this_cast = pd.Series(init[0], index=test_index_svr)
    return init[0], test_index_svr

def svr(train, test, data,n_future = 1000):
    with open('models/svr_final.pkl', 'rb') as pickle_file:
        svm = pickle.load(pickle_file)

    init = np.expand_dims(test['Close'][-30:], axis=0)

    for i in range(n_future):
        pred=np.expand_dims(svm.predict(init[:, -30:]), axis=0)
        init = np.concatenate((init, pred), axis=1)
    # print(init.shape)
    test_index_svr = []
    for i in range(int(len(data)), int(len(data))+ init.shape[1]):
        test_index_svr.append(i)

    forget_this_cast = pd.Series(init[0], index=test_index_svr)
    prev = data["Close"].values.tolist()
    return init[0], test_index_svr, prev 


# data = pd.read_csv('receivedCSV/nymex_4ind.csv')

# data


# def mape(actual, pred): 
    
#     return np.mean(np.abs((actual - pred) / actual)) * 100

def getMAE(train,test,data,n_future = 1000):
    lstm_1, test_index_01 = lstm_mae(train, test, data,n_future)
    MAE_lstm = mean_absolute_error(test['Close'], lstm_1[:len(test['Close'])])
    print("MAE for LSTM: "+str(MAE_lstm))
    
    actual, predLSTM = np.array(test['Close']), np.array(lstm_1[:len(test['Close'])])
    mapeLSTM = np.mean(np.abs((actual - predLSTM) / actual)) * 100

    stats_1, test_index1 = arima_es_mae(data,n_future)
    fc1 = pd.Series(stats_1, index=test_index1)
    MAE_stat = mean_absolute_error(test['Close'], stats_1[:len(test['Close'])])
    print("MAE for Statistical: "+str(MAE_stat))
    
    actual, predstat = np.array(test['Close']), np.array(stats_1[:len(test['Close'])])
    mapestat = np.mean(np.abs((actual - predstat) / actual)) * 100

    svr_1, test_index_svr = svr_mae(train, test, data,n_future)
    forget_this_cast = pd.Series(svr_1, index=test_index_svr)
    MAE_svm = mean_absolute_error(test['Close'], svr_1[:len(test['Close'])])
    print("MAE for SVM: "+str(MAE_svm))

    actual, predsvr = np.array(test['Close']), np.array(svr_1[:len(test['Close'])])
    mapesvr = np.mean(np.abs((actual - predsvr) / actual)) * 100

    return MAE_lstm, MAE_stat, MAE_svm, mapeLSTM, mapestat, mapesvr




# lstm_out, test_index, prev = lstm(train, test, data)

# fc1 = pd.Series(lstm_out, index=test_index)
# plt.figure(figsize=(12,5), dpi=100)
# plt.plot(train['Close'])
# plt.plot(test['Close'])
# plt.plot(fc1)
# plt.show()

# stats_out, test_index1, prev = arima_es(train, test, data)

# fc1 = pd.Series(stats_out, index=test_index1)
# plt.figure(figsize=(12,5), dpi=100)
# plt.plot(train['Close'], label='training')
# plt.plot(test['Close'], label='actual')
# plt.plot(fc1)
# plt.title('Forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()

# svr_out, test_index_svr, prev = svr(train, test, data)
# svr_out = svr_out.tolist()
# forget_this_cast = pd.Series(svr_out, index=test_index_svr)
# plt.figure(figsize=(12,5), dpi=100)
# plt.plot(train['Close'], label='training')
# plt.plot(test['Close'], label='actual')
# plt.plot(forget_this_cast, label='forecast')
# plt.title('Forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()

# print(type(stats_out[1]))
# print(len(svr_out))
# print(len(lstm_out))
# print(len(stats_out))

