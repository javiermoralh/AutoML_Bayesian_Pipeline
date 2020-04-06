# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:16:51 2020

@author: javier.moral.hernan1
"""

import numpy as np
import datawig

class Missings():

    
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    def simple_imputation(self):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()

        # Numeric
        numeric = list(X_train.select_dtypes(include=np.number).columns)
        for var in numeric:
            X_train[var].fillna(X_train[var]
                                .dropna(axis=0)
                                .median(),
                                inplace=True)
            X_test[var].fillna(X_train[var]
                               .dropna(axis=0)
                               .median(),
                               inplace=True)

        # Categorical
        categorical = [variable for variable in list(X_train.columns)
                       if variable not in numeric]
        for var in categorical:
            X_train[var].fillna(X_train[var]
                                .dropna(axis=0)
                                .mode()[0],
                                inplace=True)
            X_test[var].fillna(X_train[var]
                               .dropna(axis=0)
                               .mode()[0],
                               inplace=True)
        return X_train, X_test

    def datawig_imputation(self):

        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        external_data = self.external_data.copy()
        cols_no_missings = X_train.columns[~X_train.isnull().any()].tolist()
        cols_missings = X_train.columns[X_train.isnull().any()].tolist()
        numeric = list(X_train[cols_missings]
                       .select_dtypes(include=np.number)
                       .columns)
        categorical = [variable for variable
                       in list(X_train[cols_missings].columns)
                       if variable not in numeric]
        for col in cols_missings:
            if col in numeric:
                imputer = datawig.SimpleImputer(input_columns=cols_no_missings,
                                                output_column=col,
                                                output_path='imputer_model')
                imputer.fit(train_df=X_train, num_epochs=10)
                X_train_pred = imputer.predict(X_train.copy()).iloc[:, -1]
                mask_train = X_train[col].isnull()
                X_train.loc[mask_train, col] = X_train_pred[mask_train]
                X_test_pred = imputer.predict(X_test.copy()).iloc[:, -1]
                mask_test = X_test[col].isnull()
                X_test.loc[mask_test, col] = X_test_pred[mask_test]
            if col in categorical:
                imputer = datawig.SimpleImputer(input_columns=cols_no_missings,
                                                output_column=col,
                                                output_path='imputer_model')
                imputer.fit(train_df=X_train, num_epochs=10)
                X_train = imputer.predict(X_train.copy()).iloc[:, 0:-2]
                X_test = imputer.predict(X_test.copy()).iloc[:, 0:-2]
        return X_train, X_test, external_data

    def delete_missings(self):

        X_train = self.X_train.copy()
        X_test = self.X_test.copy()

        X_train.dropna(axis=0, inplace=True)
        X_test.dropna(axis=0, inplace=True)
        return X_train, X_test
