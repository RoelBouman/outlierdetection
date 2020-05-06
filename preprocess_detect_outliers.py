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

from pyod.models.knn import KNN 

pickle_dir = "D:\\Promotie\\formatted_OD_data"

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
from pyod.models.knn import KNN 
from pyod.models.iforest import IForest


#define parameters for each method:
abod_parameters = {"method":["fast"]}
#autoencoder_parameters = {"hidden_neurons":[[64,32,32,64]], "hidden_activation":["relu"], "output_activation":["sigmoid"], "loss":} #include more options for detailed analysis!

knn_parameters = {"n_neighbors":range(1,20)}
iforest_parameters = {"max_features":[1,2,3,4,5], "bootstrap":[True, False]}

#nested dict of methods and parameters
methods_params = {
        "ABOD": {"method":ABOD, "params":abod_parameters},
        #"AutoEncoder": {"method":AutoEncoder, "params": autoencoder_parameters}
        "KNN":{"method":KNN, "params":knn_parameters},
        "iforest":{"method":IForest, "params":iforest_parameters}
        }
    

#%% test settings:
#
# methods_params = {
#         "ABOD":{"method":ABOD, "params":abod_parameters}
#         }

methods_params = {
        "iforest":{"method":IForest, "params":iforest_parameters}
        }

# methods_params = {
#         "KNN":{"method":KNN, "params":knn_parameters}
#         }


picklefile_names = os.listdir(pickle_dir)[2:4]
#%% loop over all data

data_results = {}

for picklefile_name in picklefile_names:
    
    print(picklefile_name)
    
    #picklefile_name = "cardio.pickle"
    full_path_filename = os.path.join(pickle_dir, picklefile_name)
    
    data = pickle.load(open(full_path_filename, 'rb'))
    X, y = data["X"], np.squeeze(data["y"])
    
    #loop over all methods:
    CV_results = {}
    for method, settings in methods_params.items():
        clf = ODWrapper(settings["method"]())
        gridsearch = GridSearchCV(clf, settings["params"], scoring=scorer, cv = StratifiedKFold(n_splits=5,shuffle=True), return_train_score=False)
        gridsearch.fit(X, y)
        
        CV_results[method] = pd.DataFrame(gridsearch.cv_results_)

    data_results[picklefile_name] = CV_results




#decision_function(X) en decision_scores_ geven niet hetzelfde resultaat. => logisch, want bij decision_scores_ wordt een sample niet als zijn eigen neighbor gezien. 

#maak custom functions voor score