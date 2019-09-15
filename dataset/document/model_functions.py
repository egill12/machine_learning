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
from scipy.stats import norm
from sklearn.decomposition import PCA


def create_train_test_file(data_file, data_size, test_split, test_buffer,concat_results):
    '''
    This module will create the training and testing files to be used in the ML models.
    :param data_file: DF with price data
    param: data_size: size of the training perios
    param: test_split: size of the testing period
    param: test_buffer: the time between training and testing, to ensure no data leakage from train to test.
    param: concat_results: If True, we then use a walk forward type methodology for training and testing.
    :return: training and testing data fils.
    '''
    # Check we have enough data to train with compared to the data size
    if data_size > data_file.shape[0]:
        # Overwrite the data_length to be 90% f file, with remaining 10% as train
        data_size = int(data_file.shape[0]*0.9)
        # adding a buffer of 5 forward steps before we start trading on test data
        test_size = data_file.shape[0] - (data_size + test_buffer)
    else:
        # if a percentage test split is provided , then transform this to actual no. of points.
        if test_split <= 1:
            test_size = int(data_size*test_split)
        else:
            # if a whole number, this is accepted as the number of test points to use.
            test_size = test_split
    # training size is the first x data points and the test data is appended onto the train data
    # such that we use a walk forward testing framework
    if concat_results:
        # provide data up till the test data zone
        train_data = int(data_file.shape[0]) - (test_size + test_buffer)
        train_original = data_file.iloc[:train_data, :].reset_index(drop = True)
        # provide data in the last x points of test data available
        test_original = data_file.iloc[-test_size:, :].reset_index(drop = True)
    else:
        train_original = data_file.iloc[:int(data_size), :].reset_index(drop= True)
        test_original = data_file.iloc[int(data_size) + test_buffer: (int(data_size) + int(test_size)), :].reset_index(drop= True)
    # return two separate data files for training/testing
    return train_original, test_original

def standardise_data(dataset, full_cols, standardised_cols,window):
    '''
    This function computes the standardised returns on a rolling basis looking backwards over the window specified.
    This is the most realistic in terms of a trading strategy and also means the test data is standardised on the correct basis using the
    latest data available at each timestep.
    :param dataframe:
    :param cols:
    :return: standardised dataframe of values
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

def get_pca_features(train,test, features_to_standardise, use_pca):
    '''
    This file outputs the PCA vectors of the model to the number of features needed.
    :param data_file:
    :param model_features:
    :param output_feature:
    :return:
    '''
    pca = PCA(n_components=use_pca)
    # find the PCs
    pca = pca.fit(train[features_to_standardise])
    pca_train = pca.transform(train[features_to_standardise])
    pca_test = pca.transform(test[features_to_standardise])
    labels = ['PC%s' % i for i in range(1, use_pca + 1)]
    # add the pc values to the train and test model
    pc_number = 0
    for label in labels:
        train[label] = pd.DataFrame(pca_train[:,pc_number])
        test[label] = pd.DataFrame(pca_test[:, pc_number])
        pc_number += 1
    # Find the variance explained within each PC
    var_exp = pca.explained_variance_ratio_
    # return train , test and var explained of the pca
    return train, test, var_exp

def max_exp_sharpe(avg_sharpe, var_sharpe, ntrials):
    '''
    Link to code is from --> https://gmarti.gitlab.io/qfin/2018/05/30/deflated-sharpe-ratio.html
    :param mean_sharpe:
    :param var_sharpe:
    :param nb_trials:
    :return:
    '''
    gamma = 0.5772156649015328606
    e = np.exp(1)
    return avg_sharpe + np.sqrt(var_sharpe) * (
        (1 - gamma) * norm.ppf(1 - 1 / ntrials) + gamma * norm.ppf(1 - 1 / (ntrials * e)))

def dsr(expected_sharpe,sharpe_var,ntrials,sample_length,skew,kurtosis):
    '''
    Calculate the deflated sharpe
    :param expected_sharpe:
    :param sharpe_var:
    :param ntrials:
    :param sample_length:
    :param skew:
    :param kurtosis:
    :return:
    '''
    SR_zero = max_exp_sharpe(0, sharpe_var, ntrials)

    return norm.cdf(((expected_sharpe - SR_zero) * np.sqrt(sample_length - 1))
                    / np.sqrt(1 - skew * expected_sharpe + ((kurtosis - 1) / 4) * expected_sharpe ** 2))


def main():
    pass

if __name__ == "__main__":
    main()