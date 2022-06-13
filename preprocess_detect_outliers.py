#%% setup
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pyod.utils.utility import precision_n_scores
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import ParameterGrid
from evaluation_metrics import adjusted_precision_n_scores, average_precision, adjusted_average_precision

pickle_dir = "formatted_data"
base_result_dir = "results"
result_dir = "result_dir"
csvresult_dir = "csvresult_dir"
score_dir = "score_dir"

#picklefile_names = os.listdir(pickle_dir)

#sort picklefile_names based on size: https://stackoverflow.com/questions/20252669/get-files-from-directory-argument-sorting-by-size
# make a generator for all file paths within dirpath
all_files = ( os.path.join(basedir, filename) for basedir, dirs, files in os.walk(pickle_dir) for filename in files   )
sorted_files = sorted(all_files, key = os.path.getsize)
picklefile_names = [filename.replace(pickle_dir+os.path.sep,"") for filename in sorted_files]


#define score function:
score_functions = {"ROC/AUC": roc_auc_score, 
                   "R_precision": precision_n_scores, 
                   "adjusted_R_precision": adjusted_precision_n_scores, 
                   "average_precision": average_precision, 
                   "adjusted_average_precision": adjusted_average_precision}


verbose = True
#%% Define ensemble class for LOF

#%% Define parameter settings and methods

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN 
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
#from pyod.models.loci import LOCI #LOCI is horrendously slow. (O(n3)), aLOCI might be a decent approach, but are there implementations?
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.ecod import ECOD
#from pyod.models.sos import SOS #SOS also has memory allocation issues.
from pyod.models.combination import maximization

from additional_methods.ensemble import  Ensemble
from additional_methods.wrappers import ExtendedIForest
from additional_methods.ODIN import ODIN
from additional_methods.gen2out.gen2out import gen2Out

random_state = 1457969831 #generated using np.random.randint(0, 2**31 -1)

ensemble_LOF_krange = range(5,31)

#dict of methods and functions
method_classes = {
        "ABOD":ABOD, 
        #"CBLOF":CBLOF,
        #"u-CBLOF":CBLOF,
        # "COF":COF(n_neighbors=20, method='fast'),
        "COPOD":COPOD,
        "HBOS":HBOS,
        "kNN":KNN,
        "kth-NN":KNN,
        "IF":IForest,
        # "LMDD":LMDD(n_iter=100,dis_measure="aad", random_state=random_state),
        # "LODA":LODA(n_bins="auto"),
        "ensemble-LOF":Ensemble,
        "LOF":LOF,
        # "MCD":MCD(support_fraction=0.75, assume_centered=True, random_state=random_state),
        # "OCSVM":OCSVM(kernel="rbf", gamma="auto", nu=0.75),
        "PCA":PCA, 
        # "SOD":SOD(n_neighbors=30, ref_set=20, alpha=0.8),
        "EIF":ExtendedIForest,
        "ODIN":ODIN,
        "ECOD":ECOD
        # "gen2out":gen2Out()
        }

#dict of methods and parameters
method_parameters = {
        "ABOD":{"method":["fast"], "n_neighbors":[60]}, 
        #"CBLOF":{"n_clusters":range(2,15), "alpha":[0.7,0.8,0.9], "beta":[3,5,7], "use_weights":[True]},
        #"u-CBLOF":{"n_clusters":range(2,15), "alpha":[0.7,0.8,0.9], "beta":[3,5,7], "use_weights":[False]},
        # "COF":,
        "COPOD":{},
        "HBOS":{"n_bins":["auto"]},
        "kNN":{"n_neighbors":range(5,31), "method":["mean"]},
        "kth-NN":{"n_neighbors":range(5,31), "method":["largest"]},
        "IF":{"n_estimators":[1000], "max_samples":[128,256,512,1024]},
        # "LMDD":, #aad is the same as the MAD
        # "LODA":,
        "ensemble-LOF":{"estimators":[[LOF(n_neighbors=k) for k in ensemble_LOF_krange]], "combination_function":[maximization]},
        "LOF":{"n_neighbors":range(5,31)},
        "MCD":{"support_fraction":[0.6,0.7,0.8,0.9], "assume_centered":[True]},
        # "OCSVM": #gamma="auto"  is the same as gamma=1/d, 
         "PCA":{"n_components":[0.3,0.5,0.7,0.9]}, 
        # "SOD":,
        "EIF":{"n_estimators":[1000], "max_samples":[128,256,512,1024], "extension_level":[1,2,3]},
        "ODIN":{"n_neighbors":range(5,31)},
        "ECOD":{}
        # "gen2out":
        }


#%% loop over all data, but do not reproduce existing results


target_dir = os.path.join(base_result_dir, result_dir)
target_csvdir = os.path.join(base_result_dir, csvresult_dir)
score_csvdir = os.path.join(base_result_dir, score_dir)

if not os.path.exists(score_csvdir):
    os.makedirs(score_csvdir)
    
for picklefile_name in picklefile_names:
    
    #check if data path exists, and make it if it doesn't

    
    #print name for reporting purpose
    print("______"+picklefile_name+"______")
    
    full_path_filename = os.path.join(pickle_dir, picklefile_name)
    
    data = pickle.load(open(full_path_filename, 'rb'))
    X, y = data["X"], np.squeeze(data["y"])
                    
    
    
    #loop over all methods:

    for method_name, OD_class in method_classes.items():
        print("-" + method_name)
        hyperparameter_grid = method_parameters[method_name]
        hyperparameter_list = list(ParameterGrid(hyperparameter_grid))
        
        #loop over hyperparameter settings
        for hyperparameter_setting in hyperparameter_list:
            
            if method_name == "ensemble-LOF":
                hyperparameter_string = str(ensemble_LOF_krange)
            else:
                hyperparameter_string = str(hyperparameter_setting)
                
            if verbose:
                print(hyperparameter_string)
            
            #check whether results have  been calculated
            full_target_dir = os.path.join(target_dir, picklefile_name.replace(".pickle", ""), method_name)
            target_file_name = os.path.join(target_dir, picklefile_name.replace(".pickle", ""), method_name, hyperparameter_string+".pickle")
            if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
                if verbose:
                    print(" results already calculated, skipping recalculation")
            elif method_name == "EIF" and X.shape[1] <= hyperparameter_setting["extension_level"]:
                print("Dimensionality of dataset higher than EIF extension level, skipping...")
            else:
                
                OD_method = OD_class(**hyperparameter_setting)
                
                #Temporary fix for ECOD:
                if method_name == "ECOD" and hasattr(OD_method, "X_train"):
                    delattr(OD_method, "X_train")
                    
                pipeline = make_pipeline(RobustScaler(), OD_method)
            
                pipeline.fit(X)
                
                #correct for non pyod-like behaviour from gen2out
                if method_name == "gen2out":
                    outlier_scores = pipeline[1].decision_function(pipeline.transform(X))
                else:
                    outlier_scores = pipeline[1].decision_scores_
                
                method_performance = {method_name:{score_name: score_function(y,outlier_scores) for (score_name, score_function) in score_functions.items()}}
                method_performance_df = pd.DataFrame(method_performance).transpose()
                    
                os.makedirs(full_target_dir, exist_ok=True)
                with open(target_file_name, 'wb') as handle:
                    pickle.dump(method_performance_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                #also write csv files for easy manual inspection
                full_target_csvdir = os.path.join(target_csvdir, picklefile_name.replace(".pickle", ""), method_name)
                os.makedirs(full_target_csvdir, exist_ok=True)
                target_csvfile_name = os.path.join(full_target_csvdir, hyperparameter_string+".csv")
                method_performance_df.to_csv(target_csvfile_name)
                
                full_target_scoredir = os.path.join(score_csvdir, picklefile_name.replace(".pickle", ""), method_name)
                os.makedirs(full_target_scoredir, exist_ok=True)
                target_scorefile_name = os.path.join(full_target_scoredir, hyperparameter_string+".csv")
                np.savetxt(target_scorefile_name, outlier_scores)

