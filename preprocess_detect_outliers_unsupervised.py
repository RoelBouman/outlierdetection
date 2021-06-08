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

picklefile_names = os.listdir(pickle_dir)

#define score function:
score_functions = {"ROC/AUC": roc_auc_score, "R_precision": precision_n_scores, "adjusted_R_precision": adjusted_precision_n_scores, "average_precision": average_precision, "adjusted_average_precision": adjusted_average_precision}

#%% Define parameter settings and methods

from pyod.models.abod import ABOD
from pyod.models.cof import COF
#from pyod.models.hbos import HBOS #needs auto-histogram width selection
from pyod.models.iforest import IForest
from pyod.models.knn import KNN 
from pyod.models.lmdd import LMDD
#from pyod.models.loda import LODA #needs auto-histogram width selection
#from pyod.models.lof import LOF #first needs a robust k-choice mechanism
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.sos import SOS


random_state = 1457969831 #generated using np.random.randint(0, 2**31 -1)


#nested dict of methods and parameters
methods = {
        "ABOD":ABOD(method="fast", n_neighbors=40), 
        "COF":COF(n_neighbors=20),
        #"HBOS":,
        "kNN":KNN(n_neighbors=20,method="mean", metric="euclidean"),
        "Isolation Forest":IForest(n_estimators=1000, max_samples=256, random_state=random_state),
        "LMDD":LMDD(n_iter=100,dis_measure="aad", random_state=random_state), #aad is the same as the MAD
        #"LODA":,
        #"LOF":,
        "LOCI":LOCI(alpha=0.5, k=3), #in contrast to the paper, delta is called k in PyOD. Similarly, it uses the default of (paper notation) k=20, which cannot be altered.
        "MCD":MCD(support_fraction=0.75, assume_centered=True, random_state=random_state),
        "OCSVM":OCSVM(kernel="rbf", gamma="auto", nu=0.75), #gamma="auto"  is the same as gamma=1/d, 
        "PCA":PCA(n_components=0.5, random_state=random_state), 
        "SOD":SOD(n_neighbors=30, ref_set=20, alpha=0.8),
        "SOS":SOS(perplexity=4.5, metric="euclidean")
        }

#%% loop over all data, but do not reproduce existing results


target_dir = os.path.join(result_dir)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    
for picklefile_name in picklefile_names:
    
    #check if data path exists, and make it if it doesn't

    
    #print name for reporting purpose
    print("______"+picklefile_name+"______")
    
    full_path_filename = os.path.join(pickle_dir, picklefile_name)
    
    data = pickle.load(open(full_path_filename, 'rb'))
    X, y = data["X"], np.squeeze(data["y"])
                    
    target_file_name = os.path.join(target_dir, picklefile_name.replace(".pickle", "_results.pickle"))
    
    #check if file exists and is non-empty
    if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
        print("results already calculated, skipping recalculation")
    else:
        #loop over all methods:
        method_performance = {}
        for method_name, OD_method in methods.items():
            print(method_name)
                
            pipeline = make_pipeline(RobustScaler(), OD_method)
        
            pipeline.fit(X)
            
            outlier_scores = pipeline[1].decision_scores_
            
            method_performance[method_name] = {score_name: score_function(y,outlier_scores) for (score_name, score_function) in score_functions.items()}
        method_performance_df = pd.DataFrame(method_performance).transpose()
                
        with open(target_file_name, 'wb') as handle:
            pickle.dump(method_performance_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
