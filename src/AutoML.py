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
from src.models.stacking_module import StackingModel
warnings.filterwarnings('ignore')


class AutoML():

    def __init__(self, data, target_name):
        start_time = time.time()
        self.data = data
        self.target_name = target_name
        self.data_preprocess()
        self.stacking_creation()
        self.pipeline_building()
        self.pipeline_optimization()
        self.model_selection()
        print('Time needed: {} sec'.format(round(time.time()-start_time), 2))

    def data_preprocess(self):
        '''
        Applies DataTreatment's class functions:
        - Reads the data correctly
        - Splits the data into train and test sets
        - Applies a categorical variable encoding
        - Drops variable with constant values
        - Applies feature engineering (interactions and transformations)
        - Reduces data set memory

        Returns
        -------
        None.

        '''
        data = DataTreatment(self.data, target_name=self.target_name)
        self.X_train = data.trainX
        self.y_train = data.trainY
        self.X_test = data.testX
        self.y_test = data.testY

    def stacking_creation(self):
        '''
        Builds an stacking model based on the base classifiers and
        meta-classifier defined.

        Returns
        -------
        None.

        '''
        classifiers = ['rf', 'gbm', 'svc']
        mc = 'gbm'
        self.stacking = StackingModel(classifiers, mc)

    def pipeline_building(self):
        '''
        Designs the full pipeline to be optimized with the following steps:

        - Feature Selection
        - Predictive Model

        Returns
        -------
        None.

        '''
        self.pipeline = Pipeline(steps=[
            ('fs', FeatureSelection()), ('sclf', self.stacking.model)])

    def pipeline_optimization(self):
        '''
        Trains the full pipeline applying Bayes Optimization and Cross
        Validation in order to get the optimal hyperparameter combination.

        Returns
        -------
        None.

        '''
        (X_train_aux, X_val,
         y_train_aux, y_val) = train_test_split(
             self.X_train, self.y_train, test_size=0.25,
             random_state=42, stratify=self.y_train)
        if self.X_train.memory_usage().sum() / 1024**2 >= 4:
            bayes_opt_points = 35
            bayes_opt_iter = 50
        elif (self.X_train.memory_usage().sum() / 1024**2 > 2.5
              and self.X_train.memory_usage().sum() / 1024**2 < 4):
            bayes_opt_points = 45
            bayes_opt_iter = 70
        elif self.X_train.memory_usage().sum() / 1024**2 <= 2.5:
            bayes_opt_points = 80
            bayes_opt_iter = 120
        self.bayes_model = BayesSearchCV(
            self.pipeline, [(self.stacking.param_grid)], scoring='roc_auc',
            cv=3, refit=True,  n_jobs=-1, verbose=10, iid=True,
            return_train_score=True, n_points=bayes_opt_points,
            n_iter=bayes_opt_iter)
        self.gbm_model = XGBClassifier()
        print('Training full pipeline...')
        self.bayes_model.fit(X_train_aux, y_train_aux)

        # Select between the 3 best Bayes combination
        aux_params = pd.DataFrame(
            self.bayes_model.cv_results_).sort_values(
                by='mean_test_score', ascending=False)
        combinations = [aux_params.iloc[0]['params'],
                        aux_params.iloc[1]['params'],
                        aux_params.iloc[2]['params']]
        aucs_val = []
        for combination in combinations:
            self.pipeline.set_params(**combination)
            self.pipeline.fit(X_train_aux, y_train_aux)
            preds_comb = self.pipeline.predict(X_val)
            aucs_val.append(roc_auc_score(y_val, preds_comb))
        print(['%.4f' % elem for elem in aucs_val])
        self.best_params = aux_params['params'].iloc[
            aucs_val.index(max(aucs_val))]

    def model_selection(self):
        '''
        Selects between the trained model and Extreme Gradient Boosting
        Classifier and computes predictions on test data.

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
        self.gbm_model.fit(X_train_aux, y_train_aux)

        # Predict on validation set
        preds_stck = self.pipeline.predict(X_val)
        preds_gbm = self.gbm_model.predict(X_val)
        score_stck = roc_auc_score(y_val, preds_stck)
        score_gbm = roc_auc_score(y_val, preds_gbm)
        print('Stacking val score: {}, XGBOOST val score: {}'
              .format(round(score_stck, 4), round(score_gbm, 4)))

        # Select best model and re-train with the whole train set
        if score_stck > score_gbm:
            self.pipeline.set_params(**self.best_params)
            self.pipeline.fit(self.X_train, self.y_train)
            preds = self.pipeline.predict(self.X_test)
        else:
            self.gbm_model.fit(self.X_train, self.y_train)
            preds = self.gbm_model.predict(self.X_test)
        self.auc = roc_auc_score(self.y_test, preds)

    def prediction(self):
        '''
        Returns the best model's AUC test score.

        Returns
        -------
        None.

        '''
        print('Test prediction score {}'.format(round(self.auc, 4)))
        return self.auc
