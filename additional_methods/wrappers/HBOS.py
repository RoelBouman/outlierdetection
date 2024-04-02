#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:53:21 2023

@author: rbouman
"""

from __future__ import division
from __future__ import print_function

from  ..HBOS.hbos import HBOS
from sklearn.utils import check_array

import pandas as pd


from pyod.models.base import BaseDetector


class DynamicHBOS(BaseDetector):
    """

    """

    def __init__(self, contamination=0.1):
        super(DynamicHBOS, self).__init__(contamination=contamination)
        


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
        self._set_n_classes(y)

        self.detector_ = HBOS()
        
        self.detector_.fit(pd.DataFrame(X))
        self.decision_scores_ = self.decision_function(X)

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        

        return self.detector_.predict(pd.DataFrame(X))


    @property
    def max_samples_(self):
        """The actual number of samples.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.detector_.max_samples_