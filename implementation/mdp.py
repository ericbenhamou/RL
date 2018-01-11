# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 08:30:53 2018

@author: eric.benhamou
"""

import numpy as np
from utils import sharpe

VERBOSE =  False

class Mdp:
    def __init__(self, rl_method, r_t, L, transaction_cost):
        self.rl_method = rl_method
        self.r_t = r_t
        self.L = L
        self.transaction_cost = transaction_cost
        self.delta_cost = self.transaction_cost / 2
        self.T_max = r_t.shape[0]
        self.t = 0
        self.action = np.zeros(self.T_max )
        self.reward = np.zeros(self.T_max )
        self.capital = np.zeros(self.T_max )

    # return initial state        
    def reset(self, t0):
        self.t = t0
        self.action = np.zeros(self.T_max )
        self.reward = np.zeros(self.T_max )
        self.capital = np.zeros(self.T_max )
        self.capital = np.zeros(self.T_max )
        self.capital[:t0+1] = 1
        return self.rl_method.reset(t0, self.action)
        
    # return next state, reward
    def step(self, t, action_t):
        self.action[t] = action_t
        self.reward[t] = sharpe(action_t * self.r_t[t + 2 - self.L:t + 2]-self.delta_cost, self.L) \
            if action_t != 0 else 0
                
        self.capital[t + 1] = self.capital[t] * \
            (1 + action_t * self.r_t[t + 1]- self.delta_cost *(action_t!=0))
        
        if VERBOSE: print(t,': capital[t]',self.capital[t])
        if VERBOSE: print(t,': capital[t+1]',self.capital[t+1])
        if VERBOSE: print(t,'matrix', self.capital[t-5:t+1])
        next_state = self.rl_method.update(t, self.action)
        return (next_state, self.reward[t])
    
        
    
        
        
        