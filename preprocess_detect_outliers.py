#%% setup
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pyod.utils.utility import precision_n_scores
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from evaluation_metrics import adjusted_precision_n_scores, average_precision, adjusted_average_precision

pickle_dir = "formatted_OD_data"
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
score_functions = {"ROC/AUC": roc_auc_score, "R_precision": precision_n_scores, "adjusted_R_precision": adjusted_precision_n_scores, "average_precision": average_precision, "adjusted_average_precision": adjusted_average_precision}

#%% Define ensemble class for LOF
from sklearn.utils import check_array 
 
from pyod.models.base import BaseDetector 
from pyod.models.combination import average 
from pyod.models.lof import LOF 

class Ensemble(BaseDetector): 
     
    def __init__(self, estimators=[LOF()], combination_function=average, contamination=0.1, **kwargs): 
        super(Ensemble, self).__init__(contamination=contamination) 
        self.estimators = estimators 
        self.n_estimators_ = len(estimators) 
        self.combination_function = combination_function 
        self.kwargs = kwargs 
         
    def fit(self, X, y=None): 
        X = check_array(X) 
        n_samples = X.shape[0] 
         
        all_scores = np.zeros((n_samples,self.n_estimators_)) 
         
        for i, estimator in enumerate(self.estimators): 
            estimator.fit(X) 
            all_scores[:,i] = estimator.decision_scores_ 
             
        self.decision_scores_ = self.combination_function(all_scores, **self.kwargs) 
         
        return self 
         
    def decision_function(self, X): 
        n_samples = X.shape[0] 
         
        all_scores = np.zeros((n_samples,self.n_estimators_)) 
         
        for i, estimator in enumerate(self.estimators): 
            all_scores[:,i] = estimator.decision_function(X) 
         
        return self.combination_function(all_scores, **self.kwargs)
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

from wrappers import ExtendedIForest
from additional_methods import ODIN


random_state = 1457969831 #generated using np.random.randint(0, 2**31 -1)



#nested dict of methods and parameters
methods = {
        "ABOD":ABOD(method="fast", n_neighbors=40), 
        "CBLOF":CBLOF(n_clusters=20,use_weights=True),
        "u-CBLOF":CBLOF(n_clusters=20,use_weights=False),
        "COF":COF(n_neighbors=20, method='fast'),
        "COPOD":COPOD(),
        "HBOS":HBOS(n_bins="auto"),
        "kNN":KNN(n_neighbors=20,method="mean", metric="euclidean"),
        "Isolation Forest":IForest(n_estimators=1000, max_samples=256, random_state=random_state),
        "LMDD":LMDD(n_iter=100,dis_measure="aad", random_state=random_state), #aad is the same as the MAD
        "LODA":LODA(n_bins="auto"),
        "LOF":Ensemble(estimators=[LOF(n_neighbors=k) for k in range(10,21)], combination_function=maximization),
        "MCD":MCD(support_fraction=0.75, assume_centered=True, random_state=random_state),
        "OCSVM":OCSVM(kernel="rbf", gamma="auto", nu=0.75), #gamma="auto"  is the same as gamma=1/d, 
        "PCA":PCA(n_components=0.5, random_state=random_state), 
        "SOD":SOD(n_neighbors=30, ref_set=20, alpha=0.8),
        "EIF":ExtendedIForest(n_estimators=1000, extension_level=1),
        "ODIN":ODIN(n_neighbors=20),
        "ECOD":ECOD()
        }

#%% loop over all data, but do not reproduce existing results


#make pickle file directory
target_dir = os.path.join(result_dir)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    
#make csv directory
target_csvdir = os.path.join(csvresult_dir)

if not os.path.exists(target_csvdir):
    os.makedirs(target_csvdir)
    
    
#make outlier scores direcetory
score_csvdir = os.path.join(score_dir)

if not os.path.exists(score_csvdir):
    os.makedirs(score_csvdir)
    
for picklefile_name in picklefile_names:
    
    #check if data path exists, and make it if it doesn't

    
    #print name for reporting purpose
    print("______"+picklefile_name+"______")
    
    full_path_filename = os.path.join(pickle_dir, picklefile_name)
    
    data = pickle.load(open(full_path_filename, 'rb'))
    X, y = data["X"], np.squeeze(data["y"])
                    
    
    
    #check which files exists and are non-empty
    calculated_methods = []
    for method_name, _ in methods.items():
        target_file_name = os.path.join(target_dir, picklefile_name.replace(".pickle", "_"+method_name+"_results.pickle"))
        if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
            print(method_name + " results already calculated, skipping recalculation")
            calculated_methods.append(method_name)
            
    missing_methods = methods.copy()
    for method_name in calculated_methods:
        missing_methods.pop(method_name)
            
    #loop over all methods:

    for method_name, OD_method in missing_methods.items():
        print("starting " + method_name)
            
        #Temporary fix for ECOD:
        if method_name == "ECOD" and hasattr(OD_method, "X_train"):
            delattr(OD_method, "X_train")
            
        pipeline = make_pipeline(RobustScaler(), OD_method)
    
        pipeline.fit(X)
        
        outlier_scores = pipeline[1].decision_scores_
        
        method_performance = {method_name:{score_name: score_function(y,outlier_scores) for (score_name, score_function) in score_functions.items()}}
        method_performance_df = pd.DataFrame(method_performance).transpose()
            
        target_file_name = os.path.join(target_dir, picklefile_name.replace(".pickle", "_"+method_name+"_results.pickle"))
        with open(target_file_name, 'wb') as handle:
            pickle.dump(method_performance_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #also write csv files for easy manual inspection
        target_csvfile_name = os.path.join(target_csvdir, picklefile_name.replace(".pickle", "_"+method_name+"_results.csv"))
        method_performance_df.to_csv(target_csvfile_name)
        print("finished: " + method_name)
        
        #write scores and labels
        target_file_name = os.path.join(score_csvdir, picklefile_name.replace(".pickle", "_"+method_name+"_scores.csv"))
        np.savetxt(target_file_name, outlier_scores)
