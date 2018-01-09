# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5  17:15:44 2018

@author: eric.benhamou, david.sabbagh
"""

import numpy as np

from rl_method import Dynamic_Programming
from mdp import Mdp_Test
from context import Mdp_Context

# for reproducibility
np.random.seed(1335)  
np.set_printoptions(precision=5, suppress=True, linewidth=150)

    
mdp = Mdp_Test()
context = Mdp_Context()
rl = Dynamic_Programming(dim_states = 7,dim_actions = 2,algorithm='SARSA',learn_type='MC')
rl.initialize()
accumulated_reward = 0.
max_loop=5000
episode=1000

for i in range(max_loop):
    # varying epsilon
    epsilon = max(0.,1. - i/max_loop)

    # exploration (epsilon greedy)
    if np.random.random() > epsilon: 
        context.action = rl.best_action(mdp.state)
    else : 
        context.action= np.random.randint(0,2)

    # do temporal difference
    if i > 0 and rl.learn_type=='TD' : 
        rl.learn(context.state,context.previous_action,context.reward,context.terminal,context.next_state,context.action)
    
    # run monte carlo
    if i > 0 and rl.learn_type=='MC' :
        context.save_data()
        
        if context.terminal == 1 :
            new_rewards = rl.compute_episode_return(context.rewards)
            
            for ii in range(np.array(context.states).shape[0]):
                rl.learn(context.states[ii], context.actions[ii],new_rewards[ii],context.terminals[ii],context.next_states[ii],context.next_actions[ii])
            context.clear()
            
    # go to next state
    context.state,context.previous_action,context.reward,context.terminal,context.next_state = mdp.next_state(context.action)
    #add reward
    accumulated_reward += context.reward
    
    # diplay some results
    if i % episode == 0 : 
        print( 'iteration: ' + str(i) + ' epsilon: ' + str(epsilon))
        print( 'Q', rl.Q)
        print( 'accumulated_reward', accumulated_reward)
        accumulated_reward = 0.