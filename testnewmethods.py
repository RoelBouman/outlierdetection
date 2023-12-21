#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:23:39 2023

@author: rbouman
"""

from sklearn.datasets import load_breast_cancer

from pyod.models.cof import COF
from additional_methods.cof import COF as COF2

import numpy as np
import matplotlib.pyplot as plt

import os

from sklearn.metrics import roc_auc_score


formatted_data_dir = "formatted_data"
dataset_name = "wbc.npz"


full_path_filename = os.path.join(formatted_data_dir, dataset_name)

data  = np.load(open(full_path_filename, 'rb'))
            
X, y = data["X"], np.squeeze(data["y"])

#add duplicates to X and y:
    
X = np.concatenate([X, X[[5]*20]])
y = np.concatenate([y, y[[5]*20]])


plt.figure()
model = COF2(n_neighbors=5, method="knn")

model.fit(X)

dec_scores = model.decision_scores_

plt.hist(dec_scores)

plt.show()

print(roc_auc_score(y, dec_scores))

plt.figure()

model2 = COF2(n_neighbors=5, method="fast")

model2.fit(X)

dec_scores2 = model2.decision_scores_

plt.hist(dec_scores2)

plt.show()

print(roc_auc_score(y, dec_scores2))