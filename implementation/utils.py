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

def sharpe_lg_term(g, l):
    return np.mean(g[-l:]) / np.sqrt(np.var(g[-l:]))


def compute_episode_return(prices,r_t,action,trading_rule,L,transaction_cost):
    T_max = action.shape[0]
    equity = np.zeros(T_max )
    equity[:] = 1
    
    if trading_rule == 'daytrading':
        for t in range(T_max - 1):
            equity[t + 1] = equity[t] * (1 + action[t] * r_t[t + 1]-transaction_cost*(action[t]!=0))

    elif trading_rule == 'fixed_period':
        last_trading_time = -100
        t = 0
        while t < T_max - L:
            if action[t] != 0 and last_trading_time + L < t:
                last_trading_time = t
                equity[t + 1:t + L] = equity[t]
                equity[t + L] = equity[t] * \
                    (1 + action[t] * (prices[t + L]/prices[t]-1)-transaction_cost)
                t = t + L
            else:
                equity[t + 1] = equity[t]
                t = t + 1
        t = last_trading_time + L
        while t < T_max:
            equity[t]= equity[t - 1]
            t = t + 1

    elif trading_rule == 'hold':
        t = 1
        in_position = False
        last_trading_time = -100
        while t< T_max:
            if action[t] != action[t-1]:
                if in_position:
                    in_position = False
                    equity[t] = equity[last_trading_time] * \
                         (1 - action[t] * (prices[t]/prices[last_trading_time]-1)-transaction_cost)
                    if action[t] != 0:
                        in_position = True
                        last_trading_time = t
                else:
                    equity[t] = equity[t-1]
                    in_position = True
                    last_trading_time = t
            else:
                equity[t] = equity[t-1]
            t = t + 1
    else:
        raise ValueError('invalid trading rule: supported daytrading, fixed_period and hold')
                
    return equity