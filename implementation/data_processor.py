# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5  17:15:44 2018

@author: eric.benhamou, david.sabbagh
"""

"""
General class to load data and get back
filtered returns
"""

import pandas as pd
import numpy as np


class Data_loader:
    def __init__(self, filename, folder, remove_null_nan=True):
        dataFrame = pd.read_csv(folder + filename, delimiter=',')
        if remove_null_nan  and dataFrame['Close'].dtypes == 'object':
            dataFrame[dataFrame == 'null'] = np.nan
            dataFrame.dropna(how='any', inplace=True)
        self.dataFrame = dataFrame
        self.convert_types()

    def convert_types(self):
        float64_columns = [
            'Open',
            'High',
            'Low',
            'Close',
            'Adj Close',
            'Volume']
        datetype_columns = ['Date']
        int64_columns = ['Volume']
        for column in float64_columns:
            if self.dataFrame[column].dtype != 'float64':
                self.dataFrame[column] = pd.to_numeric(
                    self.dataFrame[column])
        for column in int64_columns:
            if self.dataFrame[column].dtype != 'int64':
                self.dataFrame[column] = pd.to_numeric(
                    self.dataFrame[column])
        for column in datetype_columns:
            self.dataFrame[column] = pd.to_datetime(self.dataFrame[column])

    def get_field(self, field_name):
        return self.dataFrame[field_name].values
