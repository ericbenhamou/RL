"""
Created on Tue Jan  5 15:11:58 2018
@author: eric.benhamou, david.sabbagh
"""

import matplotlib.pyplot as plt
import plot_utils as pu

def plot_3_ts(
        dates,
        data_1,
        data_2,
        data_3,
        x_label,
        y1_label,
        y2_label,
        y3_label,
        title, 
		save = False):
    fig, axarr = plt.subplots(3, sharex=True, sharey=False)
    axarr[0].plot(dates, data_1)
    axarr[1].plot(dates, data_2)
    axarr[2].plot(dates, data_3)
    axarr[0].set_ylabel(y1_label)
    axarr[1].set_ylabel(y2_label)
    axarr[2].set_ylabel(y3_label)
    axarr[2].set_title(title, y=-0.75, fontsize=16)
    if save: pu.save_figure(plt, title, 'compare', lgd=None)


def plot_array(dates, array, title, save = False):
    plt.figure()
    for plot_item in array:
        plt.plot(dates, plot_item)
    plt.title(title, fontsize=16)
    if save: pu.save_figure(plt, title, 'array', lgd=None)
