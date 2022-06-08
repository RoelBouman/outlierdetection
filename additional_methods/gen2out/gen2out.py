####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Email	: mengchil@cs.cmu.edu      #
####################################


import numpy as np
from scipy.spatial.distance import cityblock

from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

import time
from joblib import Parallel, delayed
from tqdm import tqdm

from .iforest import IsolationForest

from pyod.utils.utility import invert_order


class gen2Out:
	def __init__(self, lower_bound=9, upper_bound=12, max_depth=7,
				 rotate=True, contamination='auto', random_state=None):
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound
		self.max_depth = max_depth
		self.rotate = rotate
		self.contamination = contamination if contamination == 'auto' else float(contamination)
		self.random_state = random_state

	def func(self, Xs, i):
		### Fit forest with full-grown trees
		clf = IsolationForest(random_state=self.random_state,
							  max_samples=len(Xs),
							  contamination=self.contamination,
							  rotate=self.rotate).fit(Xs, max_depth=100000000)
		depths = np.mean(clf._compute_actual_depth(Xs), axis=0)
		bins = np.arange(int(depths.min()), int(depths.max() + 2))
		y, x = np.histogram(depths, bins=bins)
		return i, x[np.argmax(y)]

	def fit(self, X, y=None):
		if self.random_state:
			np.random.seed(self.random_state)
		self.n_sample = X.shape[0]

		params_arr = Parallel(n_jobs=self.upper_bound-self.lower_bound)(
								[delayed(self.func)(X[np.random.choice(self.n_sample, 2 ** i, replace=True)], i)
								for i in np.arange(self.lower_bound, self.upper_bound)])
		x_arr, y_arr = np.array(params_arr).T

		self.reg = LinearRegression(fit_intercept=False).fit(x_arr.reshape(-1, 1), y_arr)
		self.clf = IsolationForest(random_state=self.random_state,
								   max_samples=len(X),
								   contamination=self.contamination,
								   rotate=self.rotate).fit(X, max_depth=self.max_depth)

		return self

	def average_path_length(self, n):
		n = np.array(n)
		apl = self.reg.predict(np.log2([n]).T)
		apl[apl < 0] = 0
		return apl

	def decision_function(self, X):
		depths, leaves = self.clf._compute_actual_depth_leaf(X)

		new_depths = np.zeros(X.shape[0])
		for d, l in zip(depths, leaves):
			new_depths += d + self.average_path_length(l)

		scores = 2 ** (-new_depths
					   / (len(self.clf.estimators_)
						  * self.average_path_length([self.n_sample])))
		
		return invert_order(scores)

	def point_anomaly_scores(self, X):
		self = self.fit(X)
		return self.decision_function(X)
	
	def group_anomaly_scores(self, X, trials=10):
		### Fit a sequence of gen2Out0
		self.min_rate = int(np.log2(len(X)) - 8) + 1
		self.scores = np.zeros((self.min_rate, trials, len(X)))

		print('Fitting gen2Out0...')
		for i in tqdm(range(self.min_rate)):
			for j in range(trials):
				X_sampled = X[np.random.choice(len(X), int(len(X) * (1 / (2 ** i))))]
				clf = self.fit(X_sampled)
				self.scores[i][j] = clf.decision_function(X)

		### Create X-ray plot
		smax = np.max(np.mean(self.scores, axis=1), axis=0)
		self.threshold = np.mean(smax) + 3 * np.std(smax)

		sr_list = []
		xrays = np.max(np.mean(self.scores, axis=1), axis=0)
		for idx, xray in enumerate(xrays):
			if xray >= self.threshold:
				sr_list.append(idx)
		sr_list = np.array(sr_list)

		### Outlier grouping
		groups = DBSCAN().fit_predict(X[sr_list])

		self.labels = -np.ones(len(X)).astype(int)
		for idx, g in zip(sr_list, groups):
			if g != -1:
				self.labels[idx] = g + 1
				
		### Compute iso-curves
		xline = 1 / (2 ** np.arange(0, self.min_rate))
		self.s_arr = [[] for l in np.unique(self.labels) if l != -1]
		xrays_max = np.argmax(np.mean(self.scores, axis=1), axis=0)
		for idx in sr_list:
			if self.labels[idx] != -1:
				dis = cityblock([np.log2(xrays_max[idx]) / 10 + 1, xrays[idx]], [1, 1])
				self.s_arr[self.labels[idx]-1].append((2 - dis) / 2)
		
		ga_scores = np.array([np.median(s) for s in self.s_arr])
		ga_indices = [np.where(self.labels == l)[0] for l in np.unique(self.labels) if l != -1]
		
		return ga_scores, ga_indices
