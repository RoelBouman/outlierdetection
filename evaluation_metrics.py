from pyod.utils.utility import get_label_n #precision_n_scores with n=None is equal to the R-precision measure
from sklearn.utils import column_or_1d
from sklearn.metrics import precision_score, average_precision_score
import numpy as np

#copied from pyod, but changed default behaviour of precision_score warnings when y_pred is all zeroes
def precision_n_scores(y, y_pred, n=None):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.

    Returns
    -------
    precision_at_rank_n : float
        Precision at rank n score.

    """

    # turn raw prediction decision scores into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return precision_score(y, y_pred, zero_division=0)

def adjusted_precision_n_scores(y_true, y_pred, n=None):

    p_at_n = precision_n_scores(y_true, y_pred, n=n)
    
    # calculate the percentage of outliers
    if n is not None:
        outliers_fraction = n /len(y_true)
    else:
        outliers_fraction = np.count_nonzero(y_true) / len(y_true)
    
    adjusted_p_at_n = (p_at_n - outliers_fraction)/(1 - outliers_fraction)
    
    return(adjusted_p_at_n)
    
def adjusted_average_precision(y_true, y_pred):
    
    ap = average_precision_score(y_true, y_pred)
    
    # calculate the percentage of outliers
    outliers_fraction = np.count_nonzero(y_true) / len(y_true)
    
    adjusted_average_precision = (ap - outliers_fraction)/(1 - outliers_fraction)
    
    return(adjusted_average_precision)

    
    

