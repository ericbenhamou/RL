# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5  17:15:44 2018

@author: eric.benhamou, david.sabbagh
"""

import numpy as np


"""
Test class for Q and SARSA learning
"""
class Mdp_Test:
    def __init__(self):
        self.reset()
	
    '''
    Transition to the next state
    '''
    def next_state(self,action):
        prev_state = self.state
        action = action
        reward = 0.
        terminal = False
        
        if self.state < 3:
            if action == 0: 
                self.state = self.state*2+1
            else : 
                self.state = self.state*2+2
        if self.state == 3 : 
            reward = 0.
            terminal = True
        if self.state == 4 : 
            reward = -1
            terminal = True
        if self.state == 5 : 
            reward = 1
            terminal = True
        if self.state == 6 : 
            reward = 0
            terminal = True
        
        next_state = self.state
        if self.state > 2 : 
            self.reset()
        return prev_state,action,reward,terminal,next_state

    def reset(self):
        self.state = 0
        
        
        
        