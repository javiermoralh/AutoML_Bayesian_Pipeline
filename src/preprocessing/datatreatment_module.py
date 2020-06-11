# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:11:27 2020

@author: javier.moral.hernan1
"""
import numpy as np
import warnings
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import DataConversionWarning
from src.preprocessing.imputation.missings_module import Missings


class DataTreatment():
    '''
    This class preprocesses data following the mentioned steps:
        - Reads the data correctly
        - Splits the data into train and test sets
        - Drops features with constant values
        - Imputes missing values
        - Drops features with to many categories
        - Applies a categorical variable encoding
        - Scales continuous features
        - Reduces data memory

    Parameters
    ----------
    data : pandas.DataFrame
        Data set containing the explanatory and target features.
    target_name : string
        Name of the target feature.

    '''
    def __init__(self, data, target_name):
        self.targetName = target_name
        data = self.correct_target_type(data)
        data = DataTreatment.correct_decimal_separator(data)
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = self.split_data(data)
        self.drop_constant_features()
        self.impute_missing_values()
        self.check_categories()
        self.categorical_encoding()
        self.scale_features()
        self.X_train = DataTreatment.reduce_memory(self.X_train)
        self.X_test = DataTreatment.reduce_memory(self.X_test)

    def correct_target_type(self, data):
        '''
        Convert target variable from categorical to 1's and 0's

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            Input data.

        Returns
        -------
        data : TYPE pandas.DataFrame
            Input data with the target variable converted to numerical data.

        '''
        data.rename(columns=lambda x: x.strip(), inplace=True)
        data[self.targetName] = data[self.targetName].astype('category')
        data[self.targetName] = data[self.targetName].cat.codes

        return data

    @staticmethod
    def correct_decimal_separator(data):
        '''
        In case the input DataFrame has any variable with ',' as decimal
        indicator, this function tries to convert ',' into '.' and read
        the variable as float.

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            Input data.

        Returns
        -------
        data : TYPE pandas.DataFrame
            Input data with the decimal indicator corrected.

        '''
        for col in data.iloc[:, :-1].select_dtypes(exclude=np.number):
            try:
                data[col] = data[col].str.replace(
                    '.', '').str.replace(',', '.').astype('float')
            except (ValueError, AttributeError):
                pass
        return data

    def split_data(self, data):
        '''
        Split the data into Train and Test sets.

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            input data.

        Returns
        -------
        X_train : TYPE pandas.DataFrame
            features of the training sample.
        X_test : TYPE
            features of the testing sample.
        y_train : TYPE
            target of the training sample..
        y_test : TYPE
            target of the testing sample.

        '''
        X = data.drop(self.targetName, axis=1).copy()
        y = data[self.targetName].copy()
        X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=0.30, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test

    def drop_constant_features(self):
        '''
        Drop columns with constant values in training set and maps it to
        test set.

        Returns
        -------
        None.

        '''
        print('Cleaning data...')
        data2keep = (self.X_train != self.X_train.iloc[0]).any()
        self.X_train = self.X_train.loc[:, data2keep]
        self.X_test = self.X_test.loc[:, data2keep]

    def impute_missing_values(self, method='simple'):
        '''
        Impute the missing values of all variables using Missings's class
        and the method selected (simple by default).

        Returns
        -------
        None.
        '''
        print('Imputing missing...')
        imputer = Missings(self.X_train, self.X_test)
        if method == 'simple':
            (self.X_train, self.X_test) = imputer.simple_imputation()
        if method == 'datawig':
            (self.X_train, self.X_test) = imputer.datawig_imputation()
        if method == 'delete':
            (self.X_train, self.X_test) = imputer.delete_missings()

    def check_categories(self, threshold=0.5):
        '''
        Drop categorical features with too many values to be an important
        feature i.e. a categorical feature that has 500 different values in
        a dataset with 1000 rows.

        Parameters
        ----------
        threshold : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        None.

        '''
        data = self.X_train.copy()
        nrows = len(data)
        dropped = []
        for col in data.iloc[:, :-1].select_dtypes(exclude=np.number):
            if (data[col].nunique()/nrows) > threshold:
                dropped.append(col)
        data.drop(labels=dropped, axis=1, inplace=True)
        self.X_train = data
        self.X_test = self.X_test.drop(labels=dropped, axis=1)

    def categorical_encoding(self):
        '''
        Encode X_train and X_test categorical features using Catboost encoder.

        Returns
        -------
        None.

        '''
        catEncoder = ce.CatBoostEncoder(
            drop_invariant=True, return_df=True, random_state=2020)
        catEncoder.fit(self.X_train, self.y_train)
        self.X_train = catEncoder.transform(self.X_train)
        self.X_test = catEncoder.transform(self.X_test)

    def scale_features(self):
        '''
        Scale X_train and X_test numerical data using a Min-Max scaling.

        Returns
        -------
        None.

        '''
        self.numeric_features = list(
            self.X_train.select_dtypes(include=np.number).columns.values)
        warnings.filterwarnings(
            action='ignore', category=DataConversionWarning)
        scaler = MinMaxScaler()
        self.X_train[self.numeric_features] = (
            scaler.fit_transform(self.X_train[self.numeric_features]))
        self.X_test[self.numeric_features] = (
            scaler.transform(self.X_test[self.numeric_features]))
        warnings.filterwarnings(
            action='default', category=DataConversionWarning)

    @staticmethod
    def reduce_memory(df):
        '''
        Reduce X_train and X_test datasets memory by correctly limiting its
        column types.

        Parameters
        ----------
        df : TYPE pandas DataFrame
            DESCRIPTION.

        Returns
        -------
        df : TYPE pandas DataFrame
            DESCRIPTION.

        '''
        print('Reducing dataset memory...')
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if (c_min > np.iinfo(np.int8).min and
                            c_max < np.iinfo(np.int8).max):
                        df[col] = df[col].astype(np.int8)
                    elif (c_min > np.iinfo(np.int16).min and
                          c_max < np.iinfo(np.int16).max):
                        df[col] = df[col].astype(np.int16)
                    elif (c_min > np.iinfo(np.int32).min and
                          c_max < np.iinfo(np.int32).max):
                        df[col] = df[col].astype(np.int32)
                    elif (c_min > np.iinfo(np.int64).min and
                          c_max < np.iinfo(np.int64).max):
                        df[col] = df[col].astype(np.int64)
                else:
                    if (c_min > np.finfo(np.float16).min and
                            c_max < np.finfo(np.float16).max):
                        df[col] = df[col].astype(np.float16)
                    elif (c_min > np.finfo(np.float32).min and
                          c_max < np.finfo(np.float32).max):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df
