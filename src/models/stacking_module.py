# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:34:49 2019

@author: javier.moral.hernan1
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import StackingClassifier


class Stacking():
    '''
    This class creates a stacking classifier with the base models and
    meta-model setected.

    Parameters
    ----------
    classifiers : list
        List of the ensmeble base models selected.
    meta_classifier : str
        Name of the meta-model selected.

    Returns
    -------
    None.

    '''
    def __init__(self, classifiers, meta_classifier):
        print('Building Stacking Ensemble')
        self.models = {'random_forest': RandomForestClassifier(),
                       'gradient_boosting': GradientBoostingClassifier(),
                       'svc': SVC(),
                       'elastic_net': SGDClassifier(
                           penalty='elasticnet', loss = 'modified_huber'),
                       'logistic_regression': LogisticRegression(),
                       'extreme_gradient_boosting': XGBClassifier()}
        self.classifiers_list = classifiers
        self.meta_classifier = meta_classifier
        self.clfs, self.mc = self.map_models()
        self.model = self.stacking_building()

    def map_models(self):
        '''
        Map the strings of the selected models to python objects to be
        included in the Stacking.

        Returns
        -------
        classifiers_models : list
            Stacking base classifiers list.
        meta_clf : scikit-learn object
            Stacking meta-classifier.
        '''
        classifiers_models = [
            self.models[clf] for clf in self.classifiers_list]
        meta_clf = self.models[self.meta_classifier]
        return classifiers_models, meta_clf

    def stacking_building(self):
        '''
        Create the full Stacking classifier.

        Returns
        -------
        sclf : mlxtend.classifier

        '''
        # sclf = StackingCVClassifier(
        #     classifiers=self.clfs, meta_classifier=self.mc, random_state=42,  n_jobs=-1)
        sclf = StackingClassifier(
            classifiers=self.clfs, meta_classifier=self.mc, use_probas=True)
        return sclf
