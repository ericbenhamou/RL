# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:32:05 2018
@author: eric.benhamou, david.sabbagh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rl_method as rlm
from mdp import Mdp

from data_processor import Data_loader
from plot_helper import plot_3_ts, plot_array

# for reproducibility
np.random.seed(1335)
np.set_printoptions()

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
alpha = 0.05
gamma = 0.95
mean = 0.00
sigma = 0.01

transaction_cost = 0#0.0019
delta_cost = transaction_cost / 2
epsilons = [0.025, 0.05, 0.1]
epsilon = epsilons[2]
squashing_dim = 2
iterations_nb = 50

# load data
data_object = Data_loader(file, folder)
data = data_object.get_field(field)
dates = data_object.get_field('Date')
r_t = np.log(data[1:]) - np.log(data[:-1])

T_max = r_t.shape[0]

#method_type = 'Q-Learning'
method_type = 'SARSA'

random_init = False
random_init = True

episode_reset = False
episode_reset = True

adaptive_lr = False
adaptive_lr = True

rl = rlm.Rl_linear(r_t, N, M, method_type, alpha, gamma, random_init, episode_reset, adaptive_lr, mean, sigma)
rl2 = rlm.Rl_full_matrix(r_t, N, M, method_type, epsilon, alpha, gamma, random_init, episode_reset, adaptive_lr, mean, sigma)
mdp = Mdp( rl, r_t, L, transaction_cost)

# computes return
actions = []
capitals = []
t0 = max(N, L, M)

for iter in range(iterations_nb):
    state = mdp.reset(t0)
    
    for t in range(t0, T_max - 1):
        # exploration exploitation 
        action_t = np.random.randint(-1, 2) if np.random.rand() < epsilon \
            else rl.max_action(state)
        
        # update
        next_state, reward_t = mdp.step(t, action_t)
        
        # learn
        rl.learn(t, action_t, reward_t )

    actions.append(mdp.action)
    capitals.append(mdp.capital)

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
    max_plot = min(mdp.capital.shape[0],max_population)
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
plt.plot(rl.theta)
