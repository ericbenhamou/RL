# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 08:30:53 2018

@author: eric.benhamou
"""

import numpy as np
from utils import squashing_function
from abc import ABCMeta, abstractmethod


DIM_ACTIONS = 3
VERBOSE = False

'''
Abstract class for RL method
'''
class Rl_Method(metaclass=ABCMeta):
    def reset(self, t0, action):
        pass

    def update(self, t, action):
        pass

    def best_action(self):
        pass

    def next_action(self):
        pass

    def learn(self, t, action_t, reward_t):
        pass


'''
Class to handle RL method linearized
this is useful in case of high dimension state space
'''
class Rl_linear(Rl_Method):
    def __init__(self, transaction_cost, epsilon, r_t, N, M, method_type, alpha,
                 gamma, random_init, squashing_dim=2):
        if method_type != 'Q-Learning' and method_type != 'SARSA':
            raise ValueError(
                'invalid algorithm: supported Q-Learning and SARSA')
        self.transaction_cost = transaction_cost
        self.epsilon = epsilon
        self.r_t = r_t
        self.method_type = method_type
        self.N = N
        self.M = M
        self.random_init = random_init
        self.alpha = alpha
        self.gamma = gamma
        self.theta = np.zeros(self.N + 2 + self.M)
        # More advanced constant
        self.t0 = 0
        self.t = 0
        self.squashing_dim = squashing_dim
        self.__visited_states_actions = {}
        self.__initialize(r_t)

    # a set of private functions
    def __initialize(self, r_t):
        self.theta = np.random.uniform(-1.0, 1.0, self.N + 2 + self.M) \
            if self.random_init else np.zeros(self.N + 2 + self.M)
        self.states = np.zeros(self.N + 1 + self.M)
        self.next_states = np.zeros(self.N + 1 + self.M)
        self.f_t = squashing_function(r_t, a=self.squashing_dim)

    def __shift_steps(self, states, t, action, ):
        states[:-1] = states[1:]
        states[0] = 1
        states[self.N] = self.f_t[t + 1]
        states[self.N + self.M] = action[t]

    def reset(self, t0, action):
        self.t0 = t0 - 1
        self.t = t0
        self.states[0] = 1
        self.states[1:self.N + 1] = self.f_t[t0 - self.N:t0]
        self.states[self.N + 1:self.N + self.M + 1] = action[t0 - self.M - 1:t0 - 1]
        # next states
        self.next_states[0] = 1
        self.next_states[1:self.N + 1] = self.f_t[t0 - self.N + 1:t0 + 1]
        self.next_states[self.N + 1:self.N + self.M + 1] = action[t0 - self.M:t0]
        self.__visited_states_actions = {}
        self.theta = np.random.uniform(-1.0, 1.0, self.N + 2 + self.M) \
            if self.random_init else np.zeros(self.N + 2 + self.M) + np.sign(self.theta[-1])
        return self.next_states

    # return next state, reward
    def update(self, t, action):
        self.__visited_states_actions[self.__hash_state_action(action[t])] = 1
        self.__shift_steps(self.states, t - 1, action)
        self.__shift_steps(self.next_states, t, action)
        self.t = t
        return self.next_states

    # The Q linearized function
    def __Q(self, states, action_t):
        return np.sum(self.theta[:self.N + self.M + 1] *
            action_t * states) + self.theta[self.N + self.M + 1] * action_t

    # The corresponding gradient
    def __Gradient_Q(self, states, action_t):
        return np.append(states, action_t)

    def best_action(self):
        return np.sign(self.theta[-1])

    '''
    The core of the algo that makes difference between on line and off line policy
    The update rule is based on stochastic gradient ascent
    '''
    def learn(self, t, action_t, reward_t):
        if self.method_type == 'Q-Learning':
            next_action = self.best_action()
        elif self.method_type == 'SARSA':
            if np.random.rand() < self.epsilon:
                next_action = np.random.randint(-1, 2)
            else:
                next_action = self.best_action()
        else:
            raise ValueError(
                'invalid algorithm: supported Q-Learning and SARSA')
        d_k = reward_t + self.gamma * \
            self.__Q(self.next_states, next_action) - \
            self.__Q(self.states, action_t)
        learning_rate = self.alpha
        self.theta += learning_rate * d_k * \
            self.__Gradient_Q(self.states, action_t)
        self.t += 1

    def __hash_state_action(self, action):
        tmp_state = 0
        for fi in self.states:
            tmp_state = tmp_state * (self.squashing_dim + 1) + fi
        tmp_action = action + 1
        return tmp_state + tmp_action

    def next_action(self):
        next_action = self.best_action()
        if self.__hash_state_action(
                next_action) not in self.__visited_states_actions.keys():
            next_action = np.random.randint(-1, 2)
        return next_action


'''
Full matrix method
Bet method for low dimension state space
'''
class Rl_full_matrix(Rl_Method):
    def __init__(self, transaction_cost, epsilon, r_t, N, M, method_type, alpha, gamma,
                 random_init, mean, sigma, squashing_dim=2):
        if method_type != 'Q-Learning' and method_type != 'SARSA':
            raise ValueError(
                'invalid algorithm: supported Q-Learning and SARSA')
        self.method_type = method_type
        self.epsilon = epsilon
        self.transaction_cost = transaction_cost
        self.N = N
        self.M = M
        self.random_init = random_init
        self.alpha = alpha
        self.gamma = gamma
        self.state = 0
        self.next_state = 0
        # More advanced constant
        self.mean = mean
        self.sigma = sigma
        self.squashing_dim = squashing_dim
        self.q_rows = (self.squashing_dim + 1) ** self.N + \
            DIM_ACTIONS ** self.M
        self.q_cols = DIM_ACTIONS
        # initialize
        self.__initialize(r_t)

    # a set of private functions
    def __initialize(self, r_t):
        # N regular states, M past actions
        self.q_matrix = np.random.normal( self.mean, self.sigma, \
            (self.q_rows, self.q_cols)) if self.random_init else \
            np.zeros((self.q_rows, self.q_cols))
        self.N_matrix = np.zeros((self.q_rows, self.q_cols))
        self.f_t = squashing_function(r_t, a=self.squashing_dim)

    def __compute_state(self, action, t):
        tmp_state = 0
        for fi in self.f_t[t - self.N + 1:t + 1]:
            tmp_state = tmp_state * (self.squashing_dim + 1) + fi
        tmp_action = 0
        for ai in action[t - self.M:t]:
            tmp_action = tmp_action * DIM_ACTIONS + ai + 1
        return int(tmp_state + tmp_action)

    def reset(self, t0, action):
        self.t = t0
        self.N_matrix = np.zeros((self.q_rows, self.q_cols))
        self.state = self.__compute_state(action, t0 - 1)
        self.next_state = self.__compute_state(action, t0)
        return self.next_state

    # return next state, reward
    def update(self, t, action):
        self.state = self.__compute_state(action, t)
        self.next_state = self.__compute_state(action, t + 1)
        self.N_matrix[self.state, int(action[t] + 1)] += 1
        return self.next_state

    def best_action(self):
        return np.argmax(self.q_matrix[self.next_state, :]) - 1

    def learn(self, t, action_t, reward_t):
        learning_rate = self.alpha / \
            self.N_matrix[self.state, int(action_t + 1)]
        
        if self.method_type == 'Q-Learning':
            next_action = self.best_action()
        elif self.method_type == 'SARSA':
            if np.random.rand() < self.epsilon:
                next_action = np.random.randint(-1, 2)
            else:
                next_action = self.best_action()
        self.q_matrix[self.state, int(action_t + 1)] += learning_rate \
            * (reward_t + self.gamma * self.q_matrix[self.next_state, int(next_action + 1)] \
                - self.q_matrix[self.state, int(action_t + 1)])

    def next_action(self):
        next_action = self.best_action()
        if self.N_matrix[self.state, int(next_action + 1)] == 0:
            next_action = np.random.randint(-1, 2)
        return next_action
