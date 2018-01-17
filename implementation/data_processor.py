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
        data_object = pd.read_csv(folder + filename, delimiter=',')
        if remove_null_nan  and data_object['Close'].dtypes == 'object':
            data_object[data_object == 'null'] = np.nan
            data_object.dropna(how='any', inplace=True)
        self.data_object = data_object
        self.convert_types()
        self.returns = {}

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
            if self.data_object[column].dtype != 'float64':
                self.data_object[column] = pd.to_numeric(
                    self.data_object[column])
        for column in int64_columns:
            if self.data_object[column].dtype != 'int64':
                self.data_object[column] = pd.to_numeric(
                    self.data_object[column])
        for column in datetype_columns:
            self.data_object[column] = pd.to_datetime(self.data_object[column])

    def get_field(self, field_name):
        return self.data_object[field_name].values
