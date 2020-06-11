# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:34:49 2019

@author: javier.moral.hernan1
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from skopt.space import Real, Integer
# from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import StackingClassifier


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
        self.models = {'rf': RandomForestClassifier(),
                       'gbm': GradientBoostingClassifier(),
                       'svc': SVC(),
                       'elasticnet': SGDClassifier(penalty='elasticnet'),
                       'lr': LogisticRegression()}
        self.classifiers_list = classifiers
        self.meta_classifier = meta_classifier
        self.param_grid_building()
        self.clfs, self.mc = self.map_models()
        self.model = self.stacking_building()

    def param_grid_building(self):
        '''
        Builds pipeline's param grid dictionary to be appendend 
        in cross validation depending on the models selected. It also includes
        the feature selection hyperparameters.

        Returns
        -------
        None.
        '''
        param_rf = {
            'sclf__randomforestclassifier__max_depth': Integer(10, 50, None),
            'sclf__randomforestclassifier__min_samples_split':
                Integer(2, 20, None),
            'sclf__randomforestclassifier__n_estimators':
                Integer(10, 200, None),
            'sclf__randomforestclassifier__bootstrap': [True, False]}
        param_gbm = {
            'sclf__gradientboostingclassifier__max_depth':
                Integer(10, 50, None),
            'sclf__gradientboostingclassifier__validation_fraction':
                Real(0.1, 0.3, None),
            'sclf__gradientboostingclassifier__n_iter_no_change':
                Integer(1, 3, None),
            'sclf__gradientboostingclassifier__min_samples_split':
                Integer(2, 20, None),
            'sclf__gradientboostingclassifier__n_estimators':
                Integer(10, 500, None)}
        param_svc = {
            'sclf__svc__gamma': Real(1e-5, 1e-3, None),
            'sclf__svc__C': Integer(1, 10000, None)}
        param_en = {
            'sclf__sgdclassifier__alpha': Real(1e-8, 10, None),
            'sclf__sgdclassifier__l1_ratio': Real(0.0, 0.5, None)}
        clf_list = self.classifiers_list.copy()
        meta_clf = self.meta_classifier
        list_clf = clf_list + [meta_clf]
        self.param_grid = {}
        if 'rf' in list_clf:
            self.param_grid.update(param_rf)
        if 'gbm' in list_clf:
            self.param_grid.update(param_gbm)
        if 'svc' in list_clf:
            self.param_grid.update(param_svc)
        if 'elasticnet' in list_clf:
            self.param_grid.update(param_en)
        self.param_grid.update({'fs__threshold': [
            0, 0.025, 0.35, 0.05, 0.075, 0.085, 0.1, 0.125, 0.15, 0.175,
            0.2, 0.225, 0.25, 0.275, 0.3, 0.4, 0.5, 0.6, 0.7]})

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
            estimators=self.clfs, final_estimator=self.mc, n_jobs=-1, cv=3)
        return sclf
