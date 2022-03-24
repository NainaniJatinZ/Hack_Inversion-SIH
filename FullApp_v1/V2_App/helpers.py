import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.pyplot import figure
import pandas as pd
# import yfinance
# from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import time
from datetime import datetime
import math




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


def calculate_MACD(data):
    ema1 = calculate_ema(data['Close'], 12*7)
    ema2 = calculate_ema(data['Close'], 26*7)
    # len(ema1)
    what = list()
    for i in range(len(ema1)):
        what.append(ema1[i] - ema2[i])
    return what
