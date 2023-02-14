from pyod.models.base import BaseDetector
import rrcf
import numpy as np
import pandas as pd

class rrcf_wrapper():
    def __init__(self, n_trees, tree_size):
        
        self.n_trees = n_trees
        self.tree_size = tree_size
        
    # based on example batch code from: https://github.com/kLabUM/rrcf
    def fit(self, X, y=None):
         
        n = X.shape[0]
        
        tree_size = min(self.tree_size, n)
        
        forest = []
        
        if self.n_trees * tree_size < n:
            self.n_trees = np.ceil(n / tree_size) #increase n_trees if not all samples are covered.
        while len(forest) < self.n_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(n, size=(n//tree_size, tree_size), 
                                   replace=False)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
            forest.extend(trees)
        
        
        # Compute average CoDisp
        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)
        for tree in forest:
            codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)
        avg_codisp /= index
        
        self.decision_scores_ = avg_codisp