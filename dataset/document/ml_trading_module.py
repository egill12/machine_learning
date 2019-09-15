'''
author: Ed Gill

This file contains the neccessary modules for creating the training and testing files for the machine learnign algorithm.
'''

# import neccessary modules to perpare the data for entry to ML model.
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

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
        # provide data up till the test data zone
        train_data = int(data_file.shape[0]) - (test_size + test_buffer)
        train_original = data_file.iloc[:train_data, :].reset_index(drop= True)
        # provide data in the last x points of test data
        test_original = data_file.iloc[-test_size:, :].reset_index(drop= True)
    else:
        train_original = data_file.iloc[:int(data_size), :].reset_index(drop= True)  # eurusd_train.iloc[-DATA_SIZE:,:]
        test_original = data_file.iloc[int(data_size) + test_buffer: (int(data_size) + int(test_size)), :].reset_index(drop= True)
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
    true_class = [np.sign(i[0]) for i in test_target]
    return accuracy_score(true_class, predicted)

def get_scaled_returns():
    '''
    This file will scale exposure based on the next 24 hour ahead prediction
    :return: 
    '''
    pass

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

def update_performance(data_size,ntree, acc_score , information_ratio, run_time, train_date, test_date, performance_store):
    # Store the data as needed
    performance_store['data_size'].append(data_size)
    performance_store['ntree'].append(ntree)
    performance_store['Accuracy_Score'].append(acc_score)
    performance_store['Info_Ratio'].append(information_ratio)
    performance_store['run_time'].append(run_time)
    performance_store['train_date_st'].append(train_date)
    performance_store['test_date_st'].append(test_date)
    return pd.DataFrame(performance_store)


def set_params_random_forests():
    '''
    This is the control center for all the params that need to be set in the RF  modules
    :return: return all params as they have been set here
    '''
    ########################### Set Model Paramaters #############################
    param_dict = {"ntrees" : [150], "max_features" : 5, "test_buffer" : 5, "max_depth" : 30 , "data_size" : 15000 ,
                  "concat_results" : False, "test_split" : 0.25, "thold" : 0.51, "window" : 1000, "trade_horizon" : 24,
                  "use_risk_adjusted" : False , "use_binary" : False, "use_classifier" : False, "use_pca" : 0,
                  "use_separated_chunk" : False, "use_random_train_data" : True}
    # this looks back over a set period as the memory for the LSTM
      # [i for i in range(25,301,25)] # [21, 66]
    # if running pca, max features can only be same or less than the full total of features
    return param_dict


def set_params_LSTM():
    '''
    Additional params only applicable to the RF code
    return:
    '''
    lstm_dict = {'EPOCH' : 500, 'first_layer': 32, 'second_layer': 16, 'look_back' :90 }
    return lstm_dict

def initialise_process(file_location, trade_horizon, window, use_risk_adjusted, use_pca,use_binary, use_random_train_data):
    '''
    This re freshes the whole data set as needed by the ipython process
    this is the function to modify if you want different features in the model.
    :return: data_normed df with standardised values and model features to use
    '''
    data_file = pd.read_csv(file_location)  # pd.read_csv(r"/storage/eurusd_train_normed.csv")
    data_file = data_file.replace(np.nan, 0)
    ########################### Set Model Paramaters #############################
    # this looks back over a set period as the memory for the LSTM
    model_features = ["spot_v_HF", "spot_v_MF", "spot_v_LF", "HF_ema_diff",
                      "MF_ema_diff", "LF_ema_diff", "target"] #  "LDN", "NY", "Asia" removed, # target must be kept at the end
    ################### Standardise Entire Dataset using rolling lookback windows ###############
    features_to_standardise = ["spot_v_HF", "spot_v_MF", "spot_v_LF", "HF_ema_diff",
                               "MF_ema_diff", "LF_ema_diff"]
    ###### Set Targets ##############
    data_file["target"] = calculate_target(data_file, trade_horizon, use_risk_adjusted)
    # Remove infinity from the values, division by 0
    data_file["target"] = data_file["target"].replace(np.inf, 0)
    data_file['target'] = data_file["target"].replace(-np.inf, 0)
    data_file['target'] = data_file["target"].replace(np.nan, 0)
    # make it binary
    if use_binary:
        data_file['target'] = data_file["target"].apply(np.sign)
    # roughly 3 yrs of data slightly less actually
    data_normed = standardise_data(data_file, model_features, features_to_standardise, window)
    # add extra features non standardised, check we are using random or non random data
    if not use_random_train_data:
        data_normed['Date'] = data_file['Date'].iloc[window:]
    data_normed['CCY'] = data_file['CCY'].iloc[window:]
    data_normed['logret'] = data_file['logret'].iloc[window:]
    if use_pca > 0:
        # if we are using pca features, then model features need only to be PC1 and PC2 etc plus the target
        model_features = ['PC%s' %i for i in range(1,use_pca+1)]
        model_features.append("target")
    return data_normed.reset_index(drop = True), model_features, features_to_standardise

def main():
    pass

if __name__ == "__main__":
    main()
