"""
Created on Tue Jan  5 15:11:58 2018
@author: eric.benhamou, david.sabbagh
"""

#import declaration
import numpy as np
from scipy.special import expit


'''
logistic function
'''


def squashing_function(x, a=2, b=1, c=1e15, d=0, round=False):
    return a * expit(c * x - np.log(b)) - \
        d if ~round else np.round(a * expit(c * x - np.log(b)) - d)

'''
sharpe ratio over l period
'''
def sharpe(g, l):
    return g[-1] / np.sqrt(np.var(g[-l:]))
