# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:08:53 2019

@author: javier.moral.hernan1
"""
import pandas as pd
from src.AutoML import AutoML

# Dataset Examples
df = pd.read_csv("data/elephant.csv", ";")
df = pd.read_csv("data/ring.csv", ";")
df = pd.read_csv("data/yeast1.csv", ";")
df = pd.read_csv("data/no_pre_aprobados.csv", ";")
df = pd.read_csv("data/titanic.csv", ",")

# AutoML execution
automl = AutoML(data=df, target_name='Survived')
automl.get_score()


df.isnull().sum()
