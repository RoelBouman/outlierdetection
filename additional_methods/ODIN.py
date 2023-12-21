# -*- coding: utf-8 -*-
"""Outlier Detection using Indegree Number (ODIN) Algorithm
"""
# Author: Roel Bouman <roel.bouman@ru.nl>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array

from pyod.models.base import BaseDetector
from pyod.utils.utility import invert_order

import numpy as np

#Note, PREDICT is not implemented properly yet. It looks only to 1 matrix of input data at a time.
class ODIN(BaseDetector):
    """
    """
    def __init__(self, n_neighbors=20,
                 metric='minkowski', p=2, metric_params=None,
                 contamination=0.1, n_jobs=None):
        super(ODIN, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs


    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)

        self.knn_graph_  = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                            metric=self.metric,
                                            p=self.p,
                                            metric_params=self.metric_params,
                                            n_jobs=self.n_jobs,
                                            include_self=False)
        
        
        
        

        # Invert decision_scores_. Outliers comes with higher outlier scores
        self.decision_scores_ = invert_order(np.asarray(np.sum(self.knn_graph_, axis=0)).flatten())
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """TEMP
        """
        X = check_array(X)

        self.knn_graph_  = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                            metric=self.metric,
                                            p=self.p,
                                            metric_params=self.metric_params,
                                            n_jobs=self.n_jobs,
                                            include_self=False)   
        
        
        

        # Invert decision_scores_. Outliers comes with higher outlier scores
        self.decision_scores_ = invert_order(np.asarray(np.sum(self.knn_graph_, axis=0)).flatten())
        self._process_decision_scores()
