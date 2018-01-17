# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:32:05 2018
@author: eric.benhamou, david.sabbagh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rl_method as rlm
import plot_utils as pu
from mdp import Mdp

from data_processor import Data_loader
from plot_helper import plot_3_ts, plot_array
from utils import compute_episode_return

# for reproducibility
np.random.seed(1335)
np.set_printoptions()

# initial parameters
folder = 'data\\'
files = ['Generali_G.MI.csv',  # 0
         'Unicredit_UCG.MI.csv',
         'Fiat_FCA.MI.csv',
         'TelecomItalia_TI.csv',
         'Saipem_SPM.MI.csv']  # 4
file = files[3]
field = 'Close'
N = 5
L = 22
M = 1
alpha_linear = 0.05
alpha_grid = 0.05
gamma = 0.95

transaction_cost = 0.0019
epsilons = [0.025, 0.05, 0]
epsilon = epsilons[2]
squashing_dim = 2
iterations_nb = 50

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

trading_rules = [
    'daytrading0',
    'daytrading1',
    'daytrading2',
    'daytrading3',
    'daytrading4',
    'fixed_period',  # index 5
    'hold0',        # index 6
    'hold1',        # index 7
    'hold2']       # index 8
trading_rule = trading_rules[3]  # 222

# type of reinforcement learning method
transaction_cost = 0
rl1 = rlm.Rl_linear(transaction_cost, epsilon, r_t, N, M, method_type, alpha_linear, gamma, random_init)
mean = 0.00
sigma = 0.01
rl2 = rlm.Rl_full_matrix(transaction_cost, epsilon, r_t, N, M, method_type, alpha_grid, gamma, random_init, mean, sigma)
no_trade_reward = 0
mdp = Mdp(rl2, r_t, L, transaction_cost, no_trade_reward, trading_rule)

# good result: 0 RL1 SARSA Trading rule 2 no_trade_reward = 0.0019


# computes return
actions = []
equity_lines = []
start = max(N, L, M)
end = T_max - L

for iter in range(iterations_nb):
    state = mdp.reset(start)
    for t in range(start, end):
        # exploration exploitation
        if np.random.rand() < epsilon:
            action_t = np.random.randint(-1, 2)
        else:
            #action_t = mdp.rl_method.next_action()
            action_t = mdp.rl_method.best_action()
        # update
        next_state, reward_t = mdp.step(t, action_t)
        # learn
        mdp.rl_method.learn(t, action_t, reward_t)

    (equity, unused) = mdp.compute_episode_return(None )
    actions.append(mdp.action)
    equity_lines.append(equity)

# compute final action
final_action = np.zeros(actions[0].shape[0])
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
(final_capital, trades_nb) = mdp.compute_episode_return(final_action)

final_equity = []
avg_capital = 0
percent_positive = 0
for equity_i in equity_lines:
    avg_capital += equity_i[-1]
    percent_positive += equity_i[-1] > 1
    final_equity.append(equity_i[-1])
percent_positive /= iterations_nb
avg_capital /= iterations_nb

# plot result
#pu.delete_all_png_file()
plot_result = True
if plot_result:
    max_population = 100
    max_plot = min(mdp.action.shape[0], max_population)
    plot_3_ts(dates, prices, actions[0], equity_lines[0],
              'Time', 'Price', 'Action', 'Capital', 'Single MC Path')
    plot_array(dates, equity_lines[:max_plot], 'Capital')
    # plot_array(dates, actions[:max_plot], 'Actions')
    plot_3_ts(dates, prices, final_action, final_capital,
              'Time', 'Price', 'Action', 'Capital', 'Final Rules')
    plt.show()

# print final value of capital
print('final capital {:.2f}'.format(final_capital[-1]))
print('median capital {:.2f}'.format(avg_capital))
year_frac = pd.Timedelta(dates[-1] - dates[1]).days / 365.25
print('trade per year {:.2f}'.format(trades_nb / year_frac))
avg_return = final_capital[-1] ** (1 / year_frac) - 1
print('avg  return {:.2f}%'.format(avg_return * 100))
long_ret = (prices[-1] / prices[0]) ** (1 / year_frac) - 1
print('long return {:.2f}%'.format(long_ret * 100))
print('% positive = {:.2f}%'.format(percent_positive * 100))


if isinstance(mdp.rl_method, rlm.Rl_linear):
    plt.title('theta')
    plt.plot(mdp.rl_method.theta)