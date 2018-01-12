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
from utils import compute_episode_return

# for reproducibility
np.random.seed(1335)
np.set_printoptions()


# initial parameters
folder = 'data\\'
files = ['Generali_G.MI.csv',   #0
         'Unicredit_UCG.MI.csv',
         'Fiat_FCA.MI.csv',
         'TelecomItalia_TI.csv',
         'Saipem_SPM.MI.csv']   #4
file = files[3]
field = 'Close'
N = 5
M = 1
L = 22
alpha_linear = 0.005
alpha_grid = 0.05
gamma = 0.95
mean = 0.00
sigma = 0.01

transaction_cost = 0#0.0019
epsilons = [0.025, 0.05, 0.1]
epsilon = epsilons[2]
squashing_dim = 2
iterations_nb = 100

# load data
data_object = Data_loader(file, folder)
prices = data_object.get_field(field)
dates = data_object.get_field('Date')
r_t = np.log(prices[1:]) - np.log(prices[:-1])
prices = prices[1:]
dates = dates[1:]

T_max = r_t.shape[0]
r_t = r_t[:T_max]
prices = prices[:T_max]
dates = dates[:T_max]

method_type = 'Q-Learning'
#method_type = 'SARSA'
random_init = True

trading_type = 'daytrading'
#trading_type = 'fixed_period'
#trading_type = 'hold'

no_trade_reward = -1

# type of reinforcement learning method
rl1 = rlm.Rl_linear(r_t, N, M, method_type, alpha_linear, gamma, random_init, mean, sigma)
rl2 = rlm.Rl_full_matrix(r_t, N, M, method_type, alpha_grid, gamma, random_init, mean, sigma)
mdp = Mdp( rl1, r_t, L, transaction_cost, no_trade_reward, trading_type )

# computes return
actions = []
equity_lines = []
start = max(N, L, M)
end = T_max-1 if trading_type != 'fixed_period' else T_max-L

for iter in range(iterations_nb):
    state = mdp.reset(start)
    for t in range(start, end):
        # exploration exploitation 
        if np.random.rand() < epsilon:
            action_t = np.random.randint(-1, 2)
        else:
            action_t = mdp.rl_method.best_action()
        # update
        next_state, reward_t = mdp.step(t, action_t)
        # learn
        mdp.rl_method.learn(t, action_t, reward_t )

    equity = compute_episode_return(prices, r_t, mdp.action, trading_type, L, transaction_cost)
    actions.append(mdp.action)
    equity_lines.append(equity)
    

# compute final action
final_action = np.zeros(actions[0].shape[0])
for action_i in actions:
    final_action += action_i
final_action /= iterations_nb

threshold = 1 / 3
for i in range(final_action.size):
    if final_action[i] < -threshold :
        final_action[i] = -1
    elif final_action[i] < threshold :
        final_action[i] = 0
    else:
        final_action[i] = 1

# compute final capital
final_capital = compute_episode_return( prices, r_t, final_action, \
    trading_type, L, transaction_cost)

avg_capital = 0
for equity_i in equity_lines:
    avg_capital += equity_i[-L-1]
avg_capital  /= iterations_nb

# plot result
plot_result = True
if plot_result:
    max_population = 30
    max_plot = min(mdp.action.shape[0],max_population)
    plot_3_ts(dates, prices, actions[0], equity_lines[0],
              'Time', 'Price', 'Action', 'Capital', 'Single MC Path')
    plot_array(dates, equity_lines[:max_plot], 'Capital')
    plot_array(dates, actions[:max_plot], 'Actions')
    plot_3_ts(dates, prices, final_action, final_capital,
              'Time', 'Price',  'Action', 'Capital', 'Final Rules')
    plt.show()

# print final value of capital
print('final capital', final_capital[-1])
print('avg capital', avg_capital)
year_frac = pd.Timedelta( dates[-1] - dates[1]).days / 365.25
print('trade per year', final_action[final_action!=0].shape[0]/year_frac)
#plt.plot(rl.theta)
