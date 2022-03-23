import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.pyplot import figure
import pandas as pd
# import yfinance
# from mpl_finance import candlestick_ohlc
import time
from datetime import datetime
import math
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('receivedCSV/nymex_4ind.csv')

def LSTMPred(data):



    data = data.drop(columns = ['Unnamed: 0'])
    data = data.drop(columns = ['Date'])
    data = data.drop(columns = ['Open'])
    data = data.drop(columns = ['High'])
    data = data.drop(columns = ['Low'])
    data = data.drop(columns = ['DateTime'])

    data = data.fillna(data.mean())
    data.isna().sum()

    train_df,test_df = data[:1661], data[1661:] 

    train = train_df
    scaler = MinMaxScaler(feature_range=(-4,4))
    s_s = scaler.fit_transform(train['Volume'].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    train['Volume']=s_s
    test = test_df
    s_s = scaler.transform(test['Volume'].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    test['Volume']=s_s




    model_e4d4 = tf.keras.models.load_model('models/model_e4d4.h5')

    X_test = test.iloc[-50:,:]

    test.reset_index()

    init_test = np.expand_dims(X_test, axis=0)

    # init_test = np.asarray(init_test).astype(np.float32)


    for i in range(15):
        pred_init = model_e4d4.predict(init_test[:,-50:,:])
        init_test = np.concatenate((init_test, pred_init), axis = 1)

    pred_god = init_test[:,:,0].tolist()[0]

    input1 = data['Close'].values.tolist()


    test_index = []
    for i in range(0, 3054):
        test_index.append(i)
    print(len(input1),len(pred_god),len(test_index))


    x = [0 for i in range(2554)]
    pred_god = x+pred_god

    return input1,pred_god,test_index

    # # fc = pd.Series(y_god[0], index=test_index)
    # fc1 = pd.Series(pred_god, index=test_index)

    # plt.figure(figsize=(12,5), dpi=100)
    # plt.plot(input1)
    # plt.plot(fc1)
    # # plt.plot(fc)
    # plt.show()

LSTMPred(data)