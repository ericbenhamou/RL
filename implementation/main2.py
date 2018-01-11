# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:19:52 2018

@author: eric.benhamou
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:32:05 2018
@author: eric.benhamou, david.sabbagh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_processor import Data_loader
from utils import squashing_function
#from utils import sharpe
from plot_helper import plot_3_ts, plot_array


# for reproducibility
np.random.seed(1335)


def Q_func(states, action, theta):
    return np.sum(theta[:N + M + 1] * states) + theta[N + M + 1] * action

def Gradient_Q(states, action, theta):
    return np.append(states, action)


def sharpe1(g, l):
    return np.mean(g[-l:]) / np.sqrt(np.var(g[-l:]))

def sharpe2(g, l):
    return g[-1] / np.sqrt(np.var(g[-l:]))

def sharpe(g, l):
    return g[-1] / np.sqrt(np.var(g[-l:]))


# initial parameters
folder = 'data\\'
files = ['Generali_G.MI.csv',
         'Unicredit_UCG.MI.csv',
         'Fiat_FCA.MI.csv',
         'TelecomItalia_TI.csv',
         'Saipem_SPM.MI.csv']
file = 'TelecomItalia_TI.csv'
field = 'Close'
N = 5
M = 1
L = 22
lambda_e = 0.00
gamma = 0.95
mean = 0.00
sigma = 0.01

transaction_cost = 0 #0.0019
delta_cost = transaction_cost / 2
epsilons = [0.025, 0.05, 0.1]
epsilon = epsilons[2]
squashing_dim = 2
iterations_nb = 1

# load data
data_object = Data_loader(file, folder)
data = data_object.get_field(field)
dates = data_object.get_field('Date')

Q_linear = True
Q_linear = False
method_type = 'Q-Learning'
#method_type = 'SARSA'
adaptive_alpha = False
#adaptive_alpha = True
#random_initialization = False
random_initialization = True
reset_Q = False

if Q_linear:
    alpha = 0.005
else:
    alpha = 0.05
    
    
# computes return
r_t = np.log(data[1:]) - np.log(data[:-1])
f_t = squashing_function(r_t, a=squashing_dim, round=~Q_linear)
T_max = 30 #f_t.shape[0]


dim_actions = 3  # either sell, hold, buy
actions = []
capitals = []


if Q_linear:
    theta = np.random.normal(mean,sigma,N+2+M) if random_initialization else np.zeros(N + 2 + M) 
    e = np.zeros(N + 2 + M) 
    states = np.zeros(N + 1 + M)
    next_states = np.zeros(N + 1 + M)
else:
    # N regular states, M past actions
    q_matrix = np.zeros(((squashing_dim + 1) ** N + dim_actions ** M, dim_actions))

for iter in range(iterations_nb):
    if Q_linear:
        if random_initialization:
            theta = np.random.normal(theta[-1],sigma,N+2+M) 
            e = np.zeros(N + 2 + M) 
    
    # initialize theta
    action = np.zeros(T_max)
    reward = np.zeros(T_max)
    capital = np.zeros(T_max)

    # initialize
    start = max(N, L, M)
    capital[:(start + 1)] = 1

    for t in range(start, T_max - 1):
        if Q_linear:
            states[0] = 1
            states[1:N + 1] = f_t[t - N+1:t+1]
            states[N + 1:N + M + 1] = action[t - M:t]
        else:
            tmp_state = 0
            for fi in f_t[t - N+1:t+1]:
                tmp_state = tmp_state * (squashing_dim + 1) + fi
            tmp_action = 0
            for ai in action[t - M:t]:
                tmp_action = tmp_action * dim_actions + ai + 1
            state = int(tmp_state + tmp_action)

        if (np.random.rand() < epsilon):
            action[t] = np.random.randint(-1, 2)
        else:
            if Q_linear:
                action[t] = np.sign(theta[N + M + 1])
            else:
                action[t] = np.argmax(q_matrix[state, :]) - 1

        if Q_linear:
            next_states[0] = 1
            next_states[1:N + 1] = f_t[t + 2 - N:t + 2]
            next_states[N + 1:N + M + 1] = action[t - M+1:t+1]
        else:
            tmp_state = 0
            for fi in f_t[t + 2 - N:t + 2]:
                tmp_state = tmp_state * (squashing_dim + 1) + fi
            tmp_action = 0
            for ai in action[t - M+1:t+1]:
                tmp_action = tmp_action * dim_actions + ai + 1
            next_state = int(tmp_state + tmp_action)

        reward[t] = sharpe(action[t] * r_t[t + 2 - L:t + 2]-delta_cost, L) if action[t] != 0 else 0
        learning_rate = alpha / (t + 1 - start) if adaptive_alpha else alpha

        if method_type == 'Q-Learning':
            if Q_linear:
                max_action = np.sign(theta[N + M + 1])
                d_k = reward[t] + gamma * Q_func(next_states, max_action, theta) - Q_func(states, action[t], theta)
                e_k = gamma * lambda_e * e + Gradient_Q(states, action[t], theta) 
                theta += learning_rate * d_k * e_k
            else:
                max_action = np.argmax(q_matrix[next_state, :]) - 1
               
                q_matrix[state, int(action[t] + 1)] += learning_rate * (reward[t] + gamma *
                         q_matrix[next_state, max_action] - q_matrix[state, int(action[t] + 1)])

        elif method_type == 'SARSA':
            if Q_linear:
                d_k = reward[t] + gamma * Q_func(next_states, action[t], theta) - Q_func(states, action[t], theta)
                e_k = gamma * lambda_e * e + Gradient_Q(states, action[t], theta) 
                theta += learning_rate * d_k * e_k
            else:
                q_matrix[state, int(action[t] + 1)] += learning_rate * (reward[t] + gamma *
                         q_matrix[next_state, int(action[t] + 1)] - q_matrix[state, int(action[t] + 1)])
        else:
            raise ValueError(
                'invalid algorithm: supported Q-Learning and SARSA')

        capital[t + 1] = capital[t] * (1 + action[t]*r_t[t + 1]- delta_cost *(action[t]!=0))

    actions.append(action)
    capitals.append(capital)

final_action = np.zeros(T_max)
for action_i in actions:
    final_action += action_i
final_action /= iterations_nb

threshold = 1 / 3
for i in range(final_action.size):
    if final_action[i] < -threshold:
        final_action[i] = -1
    elif final_action[i] < threshold:
        final_action[i] = 0
    else:
        final_action[i] = 1

# compute final capital
final_capital = np.zeros(T_max)
final_capital[0] = 1
for t in range(T_max - 1):
    final_capital[t + 1] = final_capital[t] * (1+final_action[t]*r_t[t + 1]-delta_cost*(final_action[t]!=0))

avg_capital = 0
for cap_i in capitals:
    avg_capital += cap_i[-1]
avg_capital  /= iterations_nb


# plot result
plot_result = True

if plot_result:
    max_population = 30
    max_plot = min(capital.shape[0],max_population)
    plot_3_ts(dates[1:], data[1:], actions[0], capitals[0],
              'Time', 'Price', 'Action', 'Capital', 'Single MC Path')
    plot_array(dates[1:], capitals[:max_plot], 'Capitals')
    plot_array(dates[1:], actions[:max_plot], 'Actions')
    plot_3_ts(dates[1:], data[1:], final_action, final_capital,
              'Time', 'Price',  'Action', 'Capital', 'Final Rules')
    plt.show()

# print final value of capital
print(final_capital[-1])
print(avg_capital)
year_frac = pd.Timedelta( dates[-1] - dates[1]).days / 365.25
print(final_action[final_action!=0].shape[0]/year_frac)
if Q_linear:
    plt.plot(theta)
