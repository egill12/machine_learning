''''
This file runs the decision tree module and then returns the train test results'''
# import nsccessary modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from sklearn import tree
from model_functions import get_accuracy, erf, standardise_data, calculate_target
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics, svm

def decision_tree(train, test,use_classifier, use_risk_adjusted,ntree, max_features, max_depth):
    # take the
    X = train.iloc[:, :-1]
    X_test = test.iloc[:, :-1]
    if use_classifier:
        Y = train["target"].apply(np.sign)
        Y_test = test["target"].apply(np.sign)
    else:
        # using risk adjusted return- oontinous value
        Y = train["target"]
        Y_test = test["target"]
    # clean the data and nan values
    X = X.replace(np.nan, 0)
    Y = Y.replace(np.nan, 0)
    Y = Y.replace(np.inf, 0)
    X_test = X_test.replace(np.nan, 0)
    Y_test = Y_test.replace(np.nan, 0)
    Y_test = Y_test.replace(np.inf, 0)
    if use_classifier:
        RF = RandomForestClassifier(n_estimators=ntree, max_features= max_features,max_depth = max_depth, verbose=0)
        # clf = tree.DecisionTreeClassifier(max_leaf_nodes = 6, max_depth = 8)
    else:
        # ass in code for regresion classifier
        RF = RandomForestRegressor(n_estimators=ntree, max_features= max_features, max_depth = max_depth,verbose=0)
    RF.fit(X, Y)
    # run training on the test data
    results = RF.predict(X_test)
    # The % threshold needed to trigger a signal either way
    if use_risk_adjusted:
        acc_score = metrics.mean_squared_error(Y_test, results)  # cant get a classifier, so need to print simple mse
    else:
        acc_score = get_accuracy(Y_test, results)
    return (results, acc_score)

def backtester(results, test, trade_horizon):
    '''
    Calculate the returns of the trading strategy,and store them in the test file.
    :param results:
    :param test:
    :param trade_horizon:
    :return:
    '''
    # This needs to change to handle the change in the target
    predictions = pd.DataFrame({"Date": test['Date'], "Predictions": results})
    test_results = pd.merge(test, predictions, how="left", on="Date").fillna(0)
    # calculate the returns of the signal
    test_results["erf_signal"] = test_results['Predictions'].apply(erf)
    test_results["scaled_signal"] = test_results['Predictions'].shift(2).rolling(trade_horizon).sum() / trade_horizon
    test_results["scaled_signal_erf"] = test_results['erf_signal'].shift(2).rolling(trade_horizon).sum() / trade_horizon
    # no shift needed as we have already done that in previous step
    test_results['strat_returns'] = test_results['logret'] * test_results['scaled_signal']
    test_results['strat_returns_sum'] = test_results['strat_returns'].cumsum()
    # calculate returns for the error function
    test_results['strat_returns_erf'] = test_results['logret'] * test_results['scaled_signal_erf']
    test_results['strat_returns_sum_erf'] = test_results['strat_returns_erf'].cumsum()
    strat_return = test_results['strat_returns'].sum()
    information_ratio = (test_results['strat_returns'].mean() * 260) / (test_results['strat_returns'].std() * np.sqrt(260))
    return (test_results, strat_return, information_ratio)

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

def initialise_process(file_location, trade_horizon, window, use_risk_adjusted):
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
                      "MF_ema_diff", "LF_ema_diff", "LDN", "NY", "Asia", "target"]
    ################### Standardise Entire Dataset using rolling lookback windows ###############
    features_to_standardise = ["spot_v_HF", "spot_v_MF", "spot_v_LF", "HF_ema_diff",
                               "MF_ema_diff", "LF_ema_diff"]
    ###### Set Targets ##############
    data_file["target"] = calculate_target(data_file, trade_horizon, use_risk_adjusted)
    # Remove infinity from the values, division by 0
    data_file["target"] = data_file["target"].replace(np.inf, 0)
    data_file['target'] = data_file["target"].replace(-np.inf, 0)
    # roughly 3 yrs of data slightly less actually
    data_normed = standardise_data(data_file, model_features, features_to_standardise, window)
    # add extra features non standardised
    data_normed['Date'] = data_file['Date'].iloc[window:]
    data_normed['CCY'] = data_file['CCY'].iloc[window:]
    data_normed['logret'] = data_file['logret'].iloc[window:]
    return data_normed.reset_index(drop = False), model_features

def run_svm_model(train, test,use_classifier, use_risk_adjusted,kernel,cost):
    # This duplicates alot of code in the Dec tree module, so maybe think about removing these duplications
    X = train.iloc[:, :-1]
    X_test = test.iloc[:, :-1]
    if use_classifier:
        Y = train["target"].apply(np.sign)
        Y_test = test["target"].apply(np.sign)
    else:
        # using risk adjusted return- oontinous value
        Y = train["target"]
        Y_test = test["target"]
    # clean the data and nan values
    X = X.replace(np.nan, 0)
    Y = Y.replace(np.nan, 0)
    Y = Y.replace(np.inf, 0)
    X_test = X_test.replace(np.nan, 0)
    Y_test = Y_test.replace(np.nan, 0)
    Y_test = Y_test.replace(np.inf, 0)
    if use_classifier:
        SVM = svm.SVC(kernel=kernel, C = cost)
        # clf = tree.DecisionTreeClassifier(max_leaf_nodes = 6, max_depth = 8)
    else:
        # ass in code for regresion classifier
        SVM = svm.SVR(kernel= kernel, C = cost)
    SVM.fit(X, Y)
    # run training on the test data
    results = SVM.predict(X_test)
    # The % threshold needed to trigger a signal either way
    if use_risk_adjusted:
        acc_score = metrics.mean_squared_error(Y_test, results)  # cant get a classifier, so need to print simple mse
    else:
        acc_score = get_accuracy(Y_test, results)
    return (results, acc_score)



def main():
    '''

    :return:
    '''
    pass


if "__main__" == __name__:
        main()

