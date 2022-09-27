#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 08:56:39 2022

@author: rbouman
"""

import os
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pickle_dir = "formatted_data"
base_result_dir = "results"
result_dir = "result_dir"
csvresult_dir = "csvresult_dir"
score_dir = "score_dir"
log_dir = "logs"

method_name = "MCD"
dataset_name = "musk"



picklefile_name = dataset_name + ".pickle"


full_path_filename = os.path.join(pickle_dir, picklefile_name)

data = pickle.load(open(full_path_filename, 'rb'))
X, y = data["X"], np.squeeze(data["y"])



score_folder_path = os.path.join(base_result_dir, score_dir, dataset_name, method_name)

hyperparameter_scores = os.listdir(score_folder_path)

n_scores = len(hyperparameter_scores)

score_sums = np.zeros(y.shape)

for hyperparameter_score in hyperparameter_scores:
    full_path_filename = os.path.join(score_folder_path, hyperparameter_score)
    
    score_sums += pd.read_csv(full_path_filename, names=["scores"])["scores"]

scores = score_sums/n_scores


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

_, S, Vt = svd(X_scaled, full_matrices=False)
V = Vt.T

var_explained = S**2 / np.sum(S**2)

X_PCA = X.dot(V)
#%% make plots
plt.figure()
plt.title("class colored plot: ")

plt.scatter(X_PCA[y==0,0], X_PCA[y==0,1], label="normal")
plt.scatter(X_PCA[y==1,0], X_PCA[y==1,1], label="outlier")

plt.xlabel("PC1 " + str(var_explained[0]*100) + "% var explained")
plt.ylabel("PC2 " + str(var_explained[1]*100) + "% var explained")

plt.legend()

plt.figure()
plt.title
plt.show()

plt.figure()
plt.title("score colored plot")

plt.scatter(X_PCA[:,0], X_PCA[:,1], c=scores)
plt.colorbar()

plt.figure()
plt.title
plt.show()
