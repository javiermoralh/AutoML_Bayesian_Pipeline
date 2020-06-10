# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:32:21 2019

@author: javier.moral.hernan1
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier


class FeatureSelection(BaseEstimator, ClassifierMixin):
    '''
    This function selects the most important / predictive features of a
    dataset based in an original ensemble variable importance framework.

    Parameters
    ----------
    threshold : float
        Threshold of variable importance selected to filter features. The
        default is 0.1.

    Returns
    -------
    None.

    '''

    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def fit(self, X, y):
        '''
        Fit four models, append its feature importances into a DataFrame,
        compute the importance score for each feature averaging all
        importances and applying a cumulative sum and selects the ones above
        the thershold.

        Parameters
        ----------
        X : pandas.DataFrame
            Explanatory features data set.
        y : pandas.DataFrame
            Target feature data set.

        Returns
        -------
        importances: pandas.DataFrame
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
            importances[
                importances['importance_score'].cumsum() /
                importances['importance_score'].sum() < 1 - threshold])
        self.importances = importances
        return self

    def transform(self, X):
        '''
        Apply the fitted model and performs feature selection on the original
        dataset.

        Parameters
        ----------
        X : pandas.DataFrame
            Explanatory features data set.

        Returns
        -------
        variables_selected: pandas.DataFrame
            Data set with the selected features.
        '''
        importances = self.importances
        features_list = importances.index.values.tolist()
        variables_selected = X[features_list]
        return variables_selected
