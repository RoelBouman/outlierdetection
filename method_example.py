from preprocess_detect_outliers import preprocess_detect_outliers
#%% Define parameter settings and methods


from pyod.models.knn import KNN 

#dict of methods and functions
methods = {
        "kNN":KNN
        }

#dict of methods and parameters
method_parameters = {
        "kNN":{"n_neighbors":range(5,31), "method":["mean"]}
        }


#%% run method over all datasets

preprocess_detect_outliers(methods, method_parameters)