'''
Author: Ed Gill
This is a process to return a randomly generated series of numbers.
change alpha from -0.2 to 0.2 to move from mean reversion to strong trend.
'''

import numpy as np


def generate_trend(n_samples, alpha):
    '''

    :return: Generate a trend
    '''
   # ( range from -0.2 to 0.2 to move from mean reversion to strong trend
    trend_param = (1 / (1 - (alpha** 3)))
    x = w = np.random.normal(size=n_samples)
    for t in range(n_samples):
        x[t] = trend_param*x[t - 1] + w[t]

    return x