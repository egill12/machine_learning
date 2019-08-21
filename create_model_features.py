'''
Author: Ed Gill
This file will create the model features for the trend model.

'''
import datetime
import numpy as np

short = 5
medium = 15
long = 55
longest = 100

def trends_features(data,CCY_COL, short, medium, long, longest, medium_multiplier,long_multplier):
    '''
    This function will calculate each trend feature and return a dataframe
    :param data: data frame with all the columns as standard
    :param short:
    :param medium:
    :param long:
    :param longest:
    :param medium_multiplyer: This will cycle through the number of peridos that the medium term window uses. .i.e 24 would make everything a day
    :param long_multiplyer: at 120, long mult means we look in terms of weeks. each period is now one business week, 24*5
    :return:
    '''
    if CCY_COL not in list(data.columns):
        print("Column name not in the dataframe")
        return data
    data["logret"] = np.log(data[CCY_COL]) - np.log(data[CCY_COL].shift(1))
    # TODO: Should this be an EMA or simple average? Using EWMA now as we
    # overweight recent history
    # HF = high frequency
    data['HF_short'] = data[CCY_COL].ewm(short).mean()
    data['HF_medium'] = data[CCY_COL].ewm(medium).mean()
    data['HF_long'] = data[CCY_COL].ewm(long).mean()
    data['HF_longest'] = data[CCY_COL].ewm(longest).mean()
    # differences to spot
    data['spot_v_HF_short'] = data[CCY_COL] - data['HF_short']
    data['spot_v_HF_medium'] = data[CCY_COL] - data['HF_medium']
    data['spot_v_HF_long'] = data[CCY_COL] - data['HF_long']
    data['spot_v_HF_longest'] = data[CCY_COL] - data['HF_longest']

    # medium frequency factors, multiplyer allows us to scale up the lookback as needed.
    # days to weeks
    data['MF_short'] = data[CCY_COL].ewm(short*medium_multiplier).mean()
    data['MF_medium'] = data[CCY_COL].ewm(medium*medium_multiplier).mean()
    data['MF_long'] = data[CCY_COL].ewm(long*medium_multiplier).mean()
    data['MF_longest'] = data[CCY_COL].ewm(longest*medium_multiplier).mean()
    # differences to spot
    # to measure relative momentum
    data['spot_v_MF_short'] = data[CCY_COL] - data['MF_short']
    data['spot_v_MF_medium'] = data[CCY_COL] - data['MF_medium']
    data['spot_v_MF_long'] = data[CCY_COL] - data['MF_long']
    data['spot_v_MF_longest'] = data[CCY_COL] - data['MF_longest']
    # long term factors
    # weeks to months
    data['LF_short'] = data[CCY_COL].ewm(short*long_multplier).mean()
    data['LF_medium'] = data[CCY_COL].ewm(medium*long_multplier).mean()
    data['LF_long'] = data[CCY_COL].ewm(long*long_multplier).mean()
    data['LF_longest'] = data[CCY_COL].ewm(longest*long_multplier).mean()
    # differences to spot
    # to measure relative momentum
    data['spot_v_LF_short'] = data[CCY_COL] - data['LF_short']
    data['spot_v_LF_medium'] = data[CCY_COL] - data['LF_medium']
    data['spot_v_LF_long'] = data[CCY_COL] - data['LF_long']
    data['spot_v_LF_longest'] = data[CCY_COL] - data['LF_longest']

    # average of both spot distance and each ema distance
    # take simple average of the divergences at each time frame
    data['spot_v_HF'] = (data['spot_v_HF_short'] + data['spot_v_HF_medium'] + data['spot_v_HF_long'] + data['spot_v_HF_longest'])/4
    data['spot_v_MF'] = (data['spot_v_MF_short'] + data['spot_v_MF_medium'] + data['spot_v_MF_long'] + data['spot_v_MF_longest'])/4
    data['spot_v_LF'] = (data['spot_v_LF_short'] + data['spot_v_LF_medium'] + data['spot_v_LF_long'] + data['spot_v_LF_longest'])/4
    #differences to each ema
    # This can capture the divergences between the EMAs, which allows us to grasp the speed of the move
    data['HF_ema_diff'] = (data['HF_short']-data['HF_medium']) + (data['HF_medium']-data['HF_long']) + (data['HF_long']-data['HF_longest'])
    data['MF_ema_diff'] = (data['MF_short']-data['MF_medium']) + (data['MF_medium']-data['MF_long']) + (data['MF_long']-data['MF_longest'])
    data['LF_ema_diff'] = (data['LF_short']-data['LF_medium']) + (data['LF_medium']-data['LF_long']) + (data['LF_long']-data['LF_longest'])
    return data

def add_timezones(data):
    '''
    Timezones can exhibit trendy or mean reverting behaviour, Hence we can add this as a feature to the model.
    :return: A data frame where we have added the timezones
    '''
    # Add in hourly feature times. Think this is important as there can be certain patterns that occur into and out
    # of these time frames
    # London and NY liquid hours
    data['LDN'] = 0
    data['NY'] = 0
    data['Asia'] = 0
    # adding in timezone changes
    data['LDN'].loc[(data["timestamp"] >= datetime.time(7,0)) & (data["timestamp"] <= datetime.time(12,0))] = 1
    data['LDN'].loc[(data["timestamp"] >= datetime.time(13,0)) & (data["timestamp"] <= datetime.time(17,0))] = 0.5
    data['NY'].loc[(data["timestamp"] >= datetime.time(13,0)) & (data["timestamp"] <= datetime.time(17,0))] = 0.5
    data['NY'].loc[(data["timestamp"] >= datetime.time(18,0)) & (data["timestamp"] <= datetime.time(22,0))] = 1
    data['Asia'].loc[(data["timestamp"] >= datetime.time(23,0))] = 1
    data['Asia'].loc[(data["timestamp"] <= datetime.time(6,0))] = 1
    return data