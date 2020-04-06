# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:32:21 2019

@author: javier.moral.hernan1
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier


class FeatureSelection(BaseEstimator, ClassifierMixin):

    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def fit(self, X, y):
        '''
        This function appends the tree models feature importances into
        a DataFrame.

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            Input data.

        Returns
        -------
        importances: TYPE pandas.DataFrame
            Feature importance data.
        '''
        threshold = self.threshold
        importances = pd.DataFrame(index=list(X.columns))
        classifiers = {'rf': RandomForestClassifier(n_estimators=30),
                       'gbm': GradientBoostingClassifier(n_estimators=30),
                       'extra': ExtraTreesClassifier(n_estimators=30),
                       'xgbm': XGBClassifier(n_estimators=30)}
        for name, classifier in classifiers.items():
            clf = classifier.fit(X, y)
            clf_imp = list(clf.feature_importances_)
            importances.loc[:, name] = clf_imp
        importances['importance_score'] = (
            importances.apply(lambda x: x*0.25).sum(axis=1))
        importances.sort_values(
            by=['importance_score'], inplace=True, ascending=False)
        importances = (
            importances[importances['importance_score'].cumsum() /
            importances['importance_score'].sum() < 1 - threshold])
        self.importances = importances
        return self

    def transform(self, X):
        '''
        This function applies the fitted model and performs feature
        selection on the original dataset.

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            Input data.

        Returns
        -------
        importances: TYPE pandas.DataFrame
            Feature importance data.
        '''
        importances = self.importances
        features_list = importances.index.values.tolist()
        variables_selected = X[features_list]
        return variables_selected
