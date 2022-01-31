# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 23:59:31 2022

@author: Roel
"""
from __future__ import division
from __future__ import print_function

import eif as iso

from pyod.models.base import BaseDetector


from sklearn.utils import check_array


class ExtendedIForest(BaseDetector):
    """

    """

    def __init__(self, n_estimators=100,
                 max_samples=256,
                 contamination=0.1,
                 extension_level=1,
                 verbose=0):
        super(ExtendedIForest, self).__init__(contamination=contamination)
        
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level


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

        max_samples = min(X.shape[0], self.max_samples)
        self.detector_ = iso.iForest(X, ntrees=self.n_estimators, sample_size=max_samples, ExtensionLevel=self.extension_level)


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
        

        return self.detector_.compute_paths(X_in=X)


    @property
    def max_samples_(self):
        """The actual number of samples.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.detector_.max_samples_