# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:08:53 2019

@author: javier.moral.hernan1
"""
import pandas as pd
from src.AutoML import AutoML
# Dataset Examples
df = pd.read_csv(r"data\elephant.csv", ";")
df = pd.read_csv(r"data\ring.csv", ";")
df = pd.read_csv(r"data\yeast1.csv", ";")
df = pd.read_csv(r"data\no_pre_aprobados.csv", ";")

# AutoML execution
automl = AutoML(data=df, target_name='target')
automl.prediction()
