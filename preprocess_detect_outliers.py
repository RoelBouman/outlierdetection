#%% setup
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from pyod.utils.utility import precision_n_scores
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from pyod.models.knn import KNN 

pickle_dir = "D:\\Promotie\\outlier_detection\\formatted_OD_data"
result_dir = "D:\\Promotie\\outlier_detection\\result_dir"

picklefile_names = os.listdir(pickle_dir)

#define score function:
p_at_n_scorer = make_scorer(precision_n_scores)
roc_auc_scorer = make_scorer(roc_auc_score)

scorers = {"p@n": p_at_n_scorer, "ROC/AUC": roc_auc_scorer}
scorer_functions = {"p@n": precision_n_scores, "ROC/AUC": roc_auc_score}
#%%
def without_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}
 
class ODWrapper():
    def __init__(self, detector=KNN(), **kwargs):

        detector.set_params(**kwargs)
        self.detector=detector

    def predict(self, X,y=None):
        return self.detector.decision_function(X)
    
    def fit(self, X, y=None):
        self.detector.fit(X)
        return self
    
    def get_params(self, deep=True):
        params = self.detector.get_params(deep=deep)
        params["detector"] = self.detector
        return params
    
    def set_params(self, **parameters):
        parameters = without_keys(parameters, ["detector"])
        for parameter, value in parameters.items():
            setattr(self.detector, parameter, value)
        return self


#%% Define parameter settings and methods for grid search:

from pyod.models.abod import ABOD
#from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN 
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.sos import SOS


real_metrics = ["euclidean","cosine" ,"chebyshev" , "correlation", "rogerstanimoto", "sqeuclidean"]
# test_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'braycurtis', 'canberra', 'chebyshev',
#           'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
#           'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
#           'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
#           'sqeuclidean', 'yule']

#define parameters for each method:

abod_parameters = {"method":["fast"]}
#autoencoder_parameters = {"hidden_neurons":[[64,32,32,64]], "hidden_activation":["relu"], "output_activation":["sigmoid"], "loss":} #include more options for detailed analysis!
cblof_parameters = {"n_clusters":[8,9], "alpha":[0.8,0.9], "beta":[4,5], "use_weights":[False, True]}
cof_parameters = {"n_neighbors":[2,3]}
hbos_parameters = {"n_bins":[10,20,30], "alpha":[0.1,0.2,0.3]}
iforest_parameters = {"n_estimators":[1000], "max_samples":[0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1.0],  "max_features":[0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1.0], "bootstrap":[True, False]}
knn_parameters = {"n_neighbors":range(1,20), "method":["mean", "largest", "median"], "metric":real_metrics}
lmdd_parameters = {"n_iter":[50,100,200], "dis_measure":["aad", "var", "iqr"]}
loda_parameters = {"n_bins":[10,20,50,100,200], "random_cuts":[50,100,200,500]}
lof_parameters = {"n_neighbors":range(1,20),"metric":real_metrics}
loci_parameters = {"alpha": [0.3, 0.5, 0.7], "k":[2,3,4]}
mcd_parameters = {"support_fraction":[None]}
ocsvm_parameters = {"kernel": ["rbf", "poly", "sigmoid", "linear"]}
pca_parameters = {"n_components":[0.2,0.4,0.6,0.8,1], "whiten":[True, False]}
sod_parameters = {"n_components":[20,30,40], "ref_set":[5,10,19], "alpha":[0.2,0.4,0.6,0.8,0.9]}
sos_parameters = {"perplexity":range(1,10), "metric":real_metrics}

#nested dict of methods and parameters
methods_params = {
        "ABOD":{"method":ABOD, "params":abod_parameters},
        #"AutoEncoder":{"method":AutoEncoder, "params":autoencoder_parameters},
        "CBLOF":{"method":CBLOF, "params":cblof_parameters},
        #"COF":{"method":COF, "params":cof_parameters},
        "HBOS":{"method":HBOS, "params":hbos_parameters},
        "KNN":{"method":KNN, "params":knn_parameters},
        "iforest":{"method":IForest, "params":iforest_parameters},
        #"LMDD":{"method":LMDD, "params":lmdd_parameters},
        "LODA":{"method":LODA, "params":loda_parameters},
        "LOF":{"method":LOF, "params":lof_parameters},
        #"LOCI":{"method":LOCI, "params":loci_parameters},
        "MCD":{"method":MCD, "params":mcd_parameters},
        "OCSVM":{"method":OCSVM, "params":ocsvm_parameters},
        "PCA":{"method":PCA, "params":pca_parameters},
        #"SOD":{"method":SOD, "params":sod_parameters},
        #"SOS":{"method":SOS, "params":sos_parameters}
        }
    
   #%% test settings:
#
# methods_params = {
#         "ABOD":{"method":ABOD, "params":abod_parameters}
#         }

# methods_params = {
#         "cblof":{"method":CBLOF, "params":cblof_parameters}
#         }

# methods_params = {
#         "cof":{"method":COF, "params":cof_parameters}
#         }

# methods_params = {
#         "HBOS":{"method":HBOS, "params":hbos_parameters},
# }

# methods_params = {
#         "iforest":{"method":IForest, "params":iforest_parameters}
#         }

# methods_params = {
#         "KNN":{"method":KNN, "params":knn_parameters}
#         }

# methods_params = {
#         "LMDD":{"method":LMDD, "params":lmdd_parameters},
# }

# methods_params = {
#         "LODA":{"method":LODA, "params":loda_parameters},
# }

# methods_params = {
#         "LOF":{"method":LOF, "params":lof_parameters},
# }

# methods_params = {
#         "LOCI":{"method":LOCI, "params":loci_parameters},
# }

# methods_params = {
#         "MCD":{"method":MCD, "params":mcd_parameters},
# }

# methods_params = {
#         "OCSVM":{"method":OCSVM, "params":ocsvm_parameters},
# }

# methods_params = {
#         "PCA":{"method":PCA, "params":pca_parameters},
# }

# methods_params = {
#         "SOD":{"method":SOD, "params":sod_parameters},
# }

# methods_params = {
#         "SOS":{"method":SOS, "params":sos_parameters},
# }

#picklefile_names = os.listdir(pickle_dir)[2:4]
#%% loop over all data, but do not reproduce existing results

random_state = 1457969831 #generated using np.random.randint(0, 2**31 -1)

data_results = {}

for picklefile_name in picklefile_names:
    
    #check if data path exists, and make it if it doesn't

    
    #print name for reporting purpose
    print(picklefile_name)
    
    full_path_filename = os.path.join(pickle_dir, picklefile_name)
    
    data = pickle.load(open(full_path_filename, 'rb'))
    X, y = data["X"], np.squeeze(data["y"])
    
    #loop over all methods:
    for method, settings in methods_params.items():
        target_dir = os.path.join(result_dir, picklefile_name.replace(".pickle", ""), method)
    
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        print("______"+method)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        all_folds = [test_fold for train_fold, test_fold in skf.split(X,y)]
        for i, (inner_index, test_index) in enumerate(skf.split(X,y)):
            inner_folds = [fold for j, fold in enumerate(all_folds) if j!=i] #inner folds will be used for K-1 fold cross validation in gridsearchCV
        
            X_inner, y_inner = X[inner_index], y[inner_index]
            X_test, y_test = X[test_index], y[test_index]           
        
            print("Fold: " + str(i))
            
            target_file_name = os.path.join(target_dir,"fold_"+str(i)+".pickle")
            #check if file exists and is non-empty
            if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
                print("results already calculated, skipping recalculation")
            else:
                
                clf = ODWrapper(settings["method"]())
                pipeline = make_pipeline(RobustScaler(), clf)
                        
                clf_settings = dict()
                for key in settings["params"].keys():
                    clf_settings["odwrapper__"+key] = settings["params"][key]
            
                CV_split = [(np.concatenate(inner_folds[:k]+inner_folds[k+1:]),test_index) for k, test_index in enumerate(inner_folds)]
                
                #We need to manually refit the algorithms with the optimal hyperparameters for both metrics, this is easier than relying on refit for parts.
                gridsearch = GridSearchCV(pipeline, clf_settings, scoring=scorers, cv = CV_split, return_train_score=False, n_jobs=4, refit=False, verbose=1)
                gridsearch.fit(X, y) #CV_split ensures the folds will be handled properly, so we can pass the entire X and y matrices, we manually refit to avoid fitting parts of the data we can't use.
            
                cv_results = pd.DataFrame(gridsearch.cv_results_)
                with open(target_file_name, 'wb') as handle:
                    pickle.dump(cv_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                #Evaluate best models according to metrics on test set:
                best_results = {}
                for scorer_name in scorer_functions.keys():                        
                    best_params = cv_results["params"][cv_results["rank_test_"+scorer_name]==1].iloc[0]
                    best_clf = ODWrapper(settings["method"]())
                    best_clf.set_params(**best_params)
                    pipeline = make_pipeline(RobustScaler(), best_clf)
                    pipeline.fit(X_inner, y_inner)
                    y_test_pred = pipeline.predict(X_test)
                    score = scorer_functions[scorer_name](y_test,y_test_pred)
                    
                    best_results[scorer_name] = {"best_params":best_params, "scorer_name":scorer_name, scorer_name:score}
               
                with open(os.path.join(target_dir, "fold_"+str(i)+"_best_param_scores.pickle"), 'wb') as handle:
                    pickle.dump(best_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#maak custom functions voor score