# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:39:24 2020

@author: javier.moral.hernan1
"""
import pandas as pd
import time
import warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from src.feature_selection.feature_selection_module import FeatureSelection
from src.preprocessing.datatreatment_module import DataTreatment
from src.models.stacking_module import Stacking
from skopt.space import Real, Integer
warnings.filterwarnings('ignore')


class AutoML():
    '''
    This class ...

    Parameters
    ----------
    data : pandas.DataFrame
        Input data set to perform AutoML process.
    target_name : str
        Name of the target feature.

    Returns
    -------
    None.

    '''

    def __init__(self, data, target_name):
        start_time = time.time()
        self.data = data
        self.target_name = target_name
        self.preprocess_data()
        self.create_stacking()
        self.compile_pipeline()
        self.optimize_pipeline()
        self.select_best_model()
        print('Time needed: {} sec'.format(round(time.time()-start_time), 2))

    def preprocess_data(self):
        '''
        Apply DataTreatment's preprocessing.

        Returns
        -------
        None.

        '''
        data = DataTreatment(self.data, target_name=self.target_name)
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test

    def create_stacking(self):
        '''
        Build an stacking classifier using the base classifiers and
        meta-classifier defined.

        Returns
        -------
        None.

        '''
        self.classifiers = ['random_forest',
                            'gradient_boosting',
                            'elastic_net']
        self.mc = 'gradient_boosting'
        self.stacking = Stacking(self.classifiers, self.mc)

    def compile_pipeline(self):
        '''
        Compile the full pipeline to be optimized with the following steps:

        - Feature Selection
        - Predictive Model

        Returns
        -------
        None.

        '''
        self.pipeline = Pipeline(steps=[
            ('feature_selection', FeatureSelection()),
            ('stacking', self.stacking.model)])

    def get_param_grid(self):
        '''
        Builds pipeline's param grid dictionary to be appendend in pipeline in
        order to optimize hyperparameters using Bayes Search with cross
        validation depending on the models selected. It also includes the
        feature selection hyperparameters.

        Returns
        -------
        None.

        '''
        param_grid = {}
        for element in self.classifiers + [self.mc]:
            if element == 'random_forest':
                object_name = 'randomforestclassifier'
                param_dict = {
                    'max_depth': Integer(10, 50, None),
                    'min_samples_split': Integer(2, 20, None),
                    'n_estimators': Integer(10, 200, None),
                    'bootstrap': [True, False]}
            if element == 'gradient_boosting':
                object_name = 'gradientboostingclassifier'
                param_dict = {
                    'max_depth': Integer(10, 50, None),
                    'validation_fraction': Real(0.1, 0.3, None),
                    'n_iter_no_change': Integer(1, 3, None),
                    'min_samples_split': Integer(3, 30, None),
                    'n_estimators': Integer(10, 200, None)}
            if element == 'svc':
                object_name = 'svc'
                param_dict = {
                    'gamma': Real(1e-5, 1e-3, None),
                    'C': Integer(1, 10000, None)}
            if element == 'elastic_net':
                object_name = 'sgdclassifier'
                param_dict = {
                    'alpha': Real(1e-8, 10, None),
                    'l1_ratio': Real(0.0, 0.5, None)}
            for key, value in param_dict.items():
                pipeline_key = 'stacking__' + object_name + '__' + key
                param_grid[pipeline_key] = value
        param_grid.update({'feature_selection__threshold': Real(0, 0.9, None)})
        return param_grid

    def optimize_pipeline(self):
        '''
        Trains the full pipeline applying Bayes Optimization and Cross
        Validation in order to get the optimal hyperparameters combination.

        Returns
        -------
        None.

        '''
        (X_train_aux, X_val, y_train_aux, y_val) = train_test_split(
             self.X_train, self.y_train, test_size=0.25,
             random_state=42, stratify=self.y_train)
        param_grid = self.get_param_grid()
        bayes_model = BayesSearchCV(
            self.pipeline, param_grid, scoring='roc_auc', cv=3, refit=True,
            n_jobs=-1, verbose=10, iid=True, return_train_score=True,
            n_points=25, _iter=40)

        print('Training full pipeline...')
        bayes_model.fit(X_train_aux, y_train_aux)
        self.best_params = bayes_model.best_params_

        # Select between the 3 best Bayes combination
        # aux_params = pd.DataFrame(
        #     self.bayes_model.cv_results_).sort_values(
        #         by='mean_test_score', ascending=False)
        # combinations = [aux_params.iloc[0]['params'],
        #                 aux_params.iloc[1]['params'],
        #                 aux_params.iloc[2]['params']]
        # aucs_val = []
        # for combination in combinations:
        #     self.pipeline.set_params(**combination)
        #     self.pipeline.fit(X_train_aux, y_train_aux)
        #     preds_comb = self.pipeline.predict(X_val)
        #     aucs_val.append(roc_auc_score(y_val, preds_comb))
        # print(['%.4f' % elem for elem in aucs_val])
        # self.best_params = aux_params['params'].iloc[
        #     aucs_val.index(max(aucs_val))]

    def select_best_model(self):
        '''
        Compare the optimized pipeline with a default Extreme Gradient
        Boosting Classifier selects the one with higher AUC on validation set.

        Returns
        -------
        None.

        '''
        # Fit best model
        (X_train_aux, X_val,
         y_train_aux, y_val) = train_test_split(
             self.X_train, self.y_train, test_size=0.25,
             random_state=42, stratify=self.y_train)
        self.pipeline.set_params(**self.best_params)
        self.pipeline.fit(X_train_aux, y_train_aux)
        xgboost_model = XGBClassifier(max_depth=7, n_jobs=-1)
        xgboost_model.fit(X_train_aux, y_train_aux)

        # Predict on validation set
        preds_stck = self.pipeline.predict(X_val)
        preds_xgb = xgboost_model.predict(X_val)
        score_stck = roc_auc_score(y_val, preds_stck)
        score_xgb = roc_auc_score(y_val, preds_xgb)
        print('Stacking validation score: {}, XGBOOST validation score: {}'
              .format(round(score_stck, 4), round(score_xgb, 4)))

        # Select best model and re-train with the whole train set
        if score_stck > score_xgb:
            self.pipeline.set_params(**self.best_params)
            self.pipeline.fit(self.X_train, self.y_train)
            self.best_model = self.pipeline
        else:
            xgboost_model.fit(self.X_train, self.y_train)
            self.best_model = xgboost_model

    def get_prediction(self):
        '''
        Returns the best model's test prediction.

        Returns
        -------
        None.

        '''
        test_prediction = self.best_model.predict(self.X_test)
        return test_prediction

    def get_score(self):
        '''
        Returns the best model's AUC test score.

        Returns
        -------
        None.

        '''
        test_prediction = self.best_model.predict(self.X_test)
        auc_score = roc_auc_score(self.y_test, test_prediction)
        print('Test prediction score {}'.format(round(auc_score, 4)))
        return auc_score
