"""
Created on Tue Jan  5 15:11:58 2018
@author: eric.benhamou, david.sabbagh
"""

#import declaration
import numpy as np
import math
from scipy.special import expit


'''
logistic function
'''
VERBOSE = True


def squashing_function(x, a=2, b=1, c=1e5, d=1):
    return a * expit(c * x - np.log(b)) - d


'''
sharpe ratio over l period
'''


def sharpe(g):
    L = g.shape[0]
    if np.var(g[-L:]) > 1e-10:
        return np.mean(g[-L:]) / np.sqrt(np.var(g[-L:])) * math.sqrt(L-1)/math.sqrt(L)
    else:
        return 0


def sharpe_short_term(g):
    l = g.shape[0]
    if np.var(g[-l:]) > 1e-10:
        return g[-l] / np.sqrt(np.var(g[-l:]))
    else:
        return 0

def sharpe_short_term2(g):
    l = g.shape[0]
    if np.var(g[-l:]) > 1e-10:
        return g[-1] / np.sqrt(np.var(g[-l:]))
    else:
        return 0


def compute_episode_return(
        prices, r_t, action, trading_rule, L, transaction_cost, verbose = False):
    T_max = action.shape[0]
    equity = np.zeros(T_max)
    equity[:] = 1
    trades_nb = 0
    # handle different cases
    if trading_rule.startswith('daytrading'):
        for t in range(1, T_max - 1):
            trades_nb +=(action[t] != action[t - 1]) + \
                (action[t] * action[t - 1] == -1)
            equity[t + 1] = equity[t] * (1 + action[t] * r_t[t + 1]
               - transaction_cost *(action[t] != action[t - 1])
               - transaction_cost * (action[t] * action[t - 1] == -1))
        trades_nb = trades_nb / 2
    elif trading_rule == 'fixed_period':
        last_trading_time = -100
        t = 0
        while t < T_max - L:
            if action[t] != 0 and t >= last_trading_time + L:
                last_trading_time = t
                equity[t + 1:t + L] = equity[t]
                equity[t + L] = equity[t] * \
                    (1 + action[t] * (prices[t + L] /
                    prices[t] - 1) - 2 * transaction_cost)
                t = t + L
                trades_nb += 1
            else:
                equity[t + 1] = equity[t]
                t = t + 1
        t = last_trading_time + L
        while t < T_max:
            equity[t] = equity[t - 1]
            t = t + 1
    elif trading_rule.startswith('hold'):
        t = 1
        in_position = False
        last_trading_time = -100
        while t < T_max:
            if action[t] != action[t - 1]:
                if in_position:
                    in_position = False
                    trades_nb += 1
                    equity[t] = equity[last_trading_time] * (1 - action[t] * (
                        prices[t] / prices[last_trading_time] - 1) - 2 * transaction_cost)
                    if action[t] != 0:
                        in_position = True
                        last_trading_time = t
                else:
                    equity[t] = equity[t - 1]
                    in_position = True
                    last_trading_time = t
            else:
                equity[t] = equity[t - 1]
            t = t + 1
    else:
        raise ValueError(
            'invalid trading rule: supported daytrading, fixed_period and hold')

    return equity, trades_nb
