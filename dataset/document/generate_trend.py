'''
Author: Ed Gill
This is a process to return a randomly generated series of numbers.
change alpha from -0.2 to 0.2 to move from mean reversion to strong trend.
'''

import numpy as np
import pandas as pd
from create_model_features import trends_features
def generate_trend(n_samples, alpha, sigma):
    '''

    :return: Generate a trend
    '''
   # ( range from -0.2 to 0.2 to move from mean reversion to strong trend
    trend_param = (1 / (1 - (alpha** 3)))
    x = w = np.random.normal(size=n_samples)*sigma
    for t in range(n_samples):
        x[t] = trend_param*x[t - 1] + w[t]
    # return the trend file as a a dataframe
    trendy_ts = pd.DataFrame(x, columns = ["trend"])
    return trendy_ts

def get_trendy_data(n_samples,trend_strength,pct_stdev,CCY_COL, short, medium, long, longest, medium_multiplier,long_multplier):
    '''
    Takes the trendy series and gets the model features
    :return:
    '''
    trendy_df = generate_trend(n_samples, trend_strength, pct_stdev)
    ccy_data = trends_features(trendy_df, CCY_COL, short, medium, long, longest, medium_multiplier, long_multplier)
    ccy_data['CCY'] = trendy_df['trend']
    # need to replace log ret with simpel return
    ccy_data['logret'] = trendy_df['CCY'] - trendy_df['CCY'].shift(1)
    return ccy_data.replace(np.nan, 0)