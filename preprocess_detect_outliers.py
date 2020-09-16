#%% setup
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
#from sklearn.metrics import roc_auc_score
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
scorer = make_scorer(precision_n_scores)
#scorer = make_scorer(roc_auc_score)

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

data_results = {}

for picklefile_name in picklefile_names:
    
    #check if data path exists, and make it if it doesn't
    target_dir = os.path.join(result_dir, picklefile_name.replace(".pickle", ""))
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    #print name for reporting purpose
    print(picklefile_name)
    
    full_path_filename = os.path.join(pickle_dir, picklefile_name)
    
    data = pickle.load(open(full_path_filename, 'rb'))
    X, y = data["X"], np.squeeze(data["y"])
    
    #loop over all methods:
    CV_results = {}
    for method, settings in methods_params.items():
        
        print("______"+method)
        
        target_file_name = os.path.join(target_dir, method+".pickle")
        #check if file exists and is non-empty
        if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
            print("results already calculated, skipping recalculation")
        else:
            clf = ODWrapper(settings["method"]())
            pipeline = make_pipeline(RobustScaler(), clf)
            
            clf_settings = dict()
            for key in settings["params"].keys():
                clf_settings["odwrapper__"+key] = settings["params"][key]
            
            gridsearch = GridSearchCV(pipeline, clf_settings, scoring=scorer, cv = StratifiedKFold(n_splits=5,shuffle=True), return_train_score=False)
            gridsearch.fit(X, y)
            
            #CV_results[method] = pd.DataFrame(gridsearch.cv_results_)
            with open(target_file_name, 'wb') as handle:
                pickle.dump(pd.DataFrame(gridsearch.cv_results_), handle, protocol=pickle.HIGHEST_PROTOCOL)

#maak custom functions voor score