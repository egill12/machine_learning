'''
author: Ed Gill

This file contains the neccessary modules for creating the training and testing files for the machine learnign algorithm.
'''

# import neccessary modules to perpare the data for entry to ML model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

features_to_use = ["spot_v_HF", "spot_v_MF", "spot_v_LF", "HF_ema_diff",
                   "MF_ema_diff", "LF_ema_diff", "LDN", "NY", "Asia", "target"]

def create_train_test_file(data_file, data_size, test_split, test_buffer,concat_results):
    '''
    This module will create the traingin and testing files to be used in the ML RNN model.
    :return: training and testing data fils.
    '''
    # use a long term rolling standardisation window

    # How large should the training data be?
    if data_size > data_file.shape[0]:
        # Overwrite the data_length to be 90% f file, with remaining 10% as train
        data_size = int(data_file.shape[0]*0.9)
        # adding a buffer of 5 forward steps before we start trading on test data
        test_size = data_file.shape[0] - (data_size + test_buffer)
    else:
        if test_split < 1:
            test_size = int(data_size*test_split)
        else:
            # if a whole number, this is accepted as the number of test points to use.
            test_size = test_split
    # training size is the first x data points
    if concat_results:
        train_original = data_file.iloc[:(test_size-test_buffer), :].reset_index(drop= False)
        test_original = data_file.iloc[-test_size:, :].reset_index(drop= False)
    else:
        train_original = data_file.iloc[:int(data_size), :].reset_index(drop= False)  # eurusd_train.iloc[-DATA_SIZE:,:]
        test_original = data_file.iloc[int(data_size) + test_buffer: (int(data_size) + int(test_size)), :].reset_index(drop= False)
    return train_original , test_original

def standardise_data(dataset, full_cols, standardised_cols,window):
    '''
    This function computes the standardised returns on a rolling basis backwards.
    This si most realistic in term sof a trading strategy and also means the test data is standardised on the correct basis using the
    latest data available at each timestep.
    :param dataframe:
    :param cols:
    :return:
    '''
    train_standardised = dataset[standardised_cols].subtract(dataset[standardised_cols].rolling(window).mean())
    train_standardised = train_standardised.divide(dataset[standardised_cols].rolling(window).std())
    # we will only return the data which is outide the initial window standardisation period
    # add non standardised features
    for feature in full_cols:
        if feature not in standardised_cols:
            train_standardised[feature] = dataset[feature]
    # This function now returns the neccessary file with both standard and non standardised columns.
    return train_standardised.loc[window:,:]

def calculate_target(data_df,trade_horizon,use_risk_adjusted):
    '''
    Take the raw dataseries of log returns.
    :param data_df:
    :param horizon:
    :param use_risk_adjusted:
    :return:  return the risk adjusted return or the raw percent ahead return
    '''
    # Using a shift = 2 so that the forward return starts from exactly the next future time step.
    if use_risk_adjusted:
        return data_df['logret'].iloc[::-1].shift(2).rolling(trade_horizon).sum().values[::-1]/data_df['logret'].iloc[::-1].shift(2).rolling(trade_horizon).std().values[::-1]
    else:
        return data_df['logret'].iloc[::-1].shift(2).rolling(trade_horizon).sum().values[::-1]



def create_dataset(dataset, populate_target, look_back, test):
    '''
    This creates the data for  passing to the LSTM module
    :param dataset:
    :param populate_target:
    :param look_back:
    :return:
    '''
    dataX, dataY, target_dates = [], [], []
    for i in range(len(dataset) - look_back + 1):
        # this takes the very last col as the target
        a = dataset[i:(i + look_back), :-1]
        dataX.append(a)
        # this code assumes that the target vector is the very last col.
        dataY.append(dataset[i + look_back - 1, -1])
        if populate_target:
            target_dates.append(test['Date'].loc[i + look_back - 1])
    return np.array(dataX), np.array(dataY), target_dates

def signal(output, thold):
    '''
    :param x: Create a signal from the predicted softmax activation output
    :return: signal to trade
    '''
    if output >= thold:
        return 1
    elif output <= (1-thold):
        return -1
    else:
        return 0

def get_accuracy(predicted, test_target ):
    '''
    :return: the prediction accuracy of our model
    '''
    return accuracy_score(test_target, predicted)

def get_scaled_returns():
    '''
    This file will scale exposure based on the next 24 hour ahead prediction
    :return:
    '''
    pass

def erf(row_value):
    ''''
    This applies the error function to smoooth the risk adjusted return prediction
    mapps all numbers from -1 to + 1
    '''
    return (2*(1/(1 + np.exp(-row_value)))-1)

def get_total_data_needed(test_split, data_size,test_buffer):
    '''

    :param test_split:
    :return: the data size needed for all computations
    '''
    if test_split <1 :
        return int(data_size*(1 + test_split)) + test_buffer
    else:
        return int(data_size + test_split + test_buffer)

def main():
    pass

if __name__ == "__main__":
    main()
