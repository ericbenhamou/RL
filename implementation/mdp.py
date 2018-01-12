# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 08:30:53 2018

@author: eric.benhamou
"""

VERBOSE = True

import numpy as np
from utils import sharpe_lg_term, sharpe, compute_episode_return

class Mdp:
    def __init__(self, rl_method, r_t, L, transaction_cost, no_trade_reward, trading_rule):
        self.rl_method = rl_method
        self.r_t = r_t
        self.L = L
        self.transaction_cost = transaction_cost
        self.delta_cost = transaction_cost/2
        self.T_max = r_t.shape[0]
        self.action = np.zeros(self.T_max )
        self.reward = np.zeros(self.T_max )
        self.trading_rule = trading_rule
        self.no_trade_reward = no_trade_reward
        self.last_position_time = -self.L -1
        self.index_prices = np.zeros(self.T_max)
        self.index_prices[0] = r_t[0]
        for t in range(1, r_t.shape[0]):
            self.index_prices[t]=self.index_prices[t-1]*r_t[t]

    # return initial state        
    def reset(self, t0):
        self.action = np.zeros(self.T_max )
        self.reward = np.zeros(self.T_max )
        self.last_position_time = -self.L -1
        return self.rl_method.reset(t0, self.action)
        
    # return next state, reward
    def step(self, t, action_t):
        self.action[t] = action_t
        
        if self.trading_rule == 'daytrading':
            if action_t != 0 :
                self.reward[t] = sharpe(action_t * self.r_t[t + 2 - self.L:t + 2] - self.transaction_cost, self.L) 
            else:
                self.reward[t] = self.no_trade_reward
        elif self.trading_rule == 'fixed_period':
            if action_t != 0 and self.last_position_time + self.L < t:
                self.reward[t] = sharpe_lg_term(action_t * self.r_t[t + 1:t + self.L + 1] - self.transaction_cost/self.L, self.L) 
                self.last_position_time = t
            else:
                self.reward[t] = self.no_trade_reward
                
        elif self.trading_rule == 'hold':
            if self.action[t] != self.action[t-1] and self.action[t] != 0:
                self.reward[t] = sharpe_lg_term( self.action[t] * self.r_t[t + 1:t + self.L + 1] - self.transaction_cost/self.L, self.L)
            else:
                self.reward[t] = self.no_trade_reward
        else:
            raise ValueError('invalid trading rule: supported daytrading, fixed_period and hold')
                
        next_state = self.rl_method.update(t, self.action)
        return (next_state, self.reward[t])
    
    
    def compute_episode_return(self, action = None ):
        return compute_episode_return( self.index_prices, self.r_t, \
                self.action if action is None else action, \
                self.trading_rule, self.L, self.transaction_cost)
