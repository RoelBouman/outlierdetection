#%% setup
import pickle
import os
import gc
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pyod.utils.utility import precision_n_scores
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import ParameterGrid
from evaluation_metrics import adjusted_precision_n_scores, average_precision, adjusted_average_precision

import argparse

pickle_dir = "formatted_data"
base_result_dir = "results"
result_dir = "result_dir"
csvresult_dir = "csvresult_dir"
score_dir = "score_dir"
log_dir = "logs"
preprocessed_data_dir = "preprocessed_data"
DeepSVDD_dir = "additional_methods/Deep-SVDD"

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


#%% argument parsing for command line functionality
# Create the parser
arg_parser = argparse.ArgumentParser(description='Run selected methods over all datasets')

# Add the arguments
arg_parser.add_argument('--method',
                       metavar='M',
                       dest='method',
                       default='all',
                       type=str,
                       help='The method that you would like to run')

arg_parser.add_argument('--verbose',
                       metavar='V',
                       dest='verbose',
                       default=1,
                       type=int,
                       help='The verbosity of the pipeline execution.')

# Execute the parse_args() method
parsed_args = arg_parser.parse_args()

method_to_run = parsed_args.method
verbose = parsed_args.verbose

#%% Define parameter settings and methods

from pyod.models.rgraph import RGraph
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.gmm import GMM

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
from pyod.models.ecod import ECOD
#from pyod.models.sos import SOS #SOS also has memory allocation issues.
from pyod.models.combination import maximization

from additional_methods.ensemble import  Ensemble
from additional_methods.wrappers.ExtendedIForest import ExtendedIForest
from additional_methods.ODIN import ODIN
#from additional_methods.gen2out.gen2out import gen2Out

from additional_methods.wrappers.AE import AE_wrapper
from additional_methods.wrappers.VAE import VAE_wrapper
from additional_methods.wrappers.AnoGAN import AnoGAN_wrapper

ensemble_LOF_krange = range(5,31)

#dict of methods and functions
method_classes = {
        "RGraph":RGraph,
        "INNE":INNE,
        "GMM":GMM,
        "KDE":KDE,
        "ABOD":ABOD, 
        "CBLOF":CBLOF,
        "u-CBLOF":CBLOF,
        "COF":COF,
        "COPOD":COPOD,
        "HBOS":HBOS,
        "kNN":KNN,
        "kth-NN":KNN,
        "IF":IForest,
        "LMDD":LMDD,
        "LODA":LODA,
        "ensemble-LOF":Ensemble,
        "LOF":LOF,
        "MCD":MCD,
        "OCSVM":OCSVM,
        "PCA":PCA, 
        "EIF":ExtendedIForest,
        "ODIN":ODIN,
        "ECOD":ECOD,
        # "gen2out":gen2Out()
        "AE":AE_wrapper,
        "VAE":VAE_wrapper,
        "beta-VAE":VAE_wrapper,
        "AnoGAN":AnoGAN_wrapper
        }

#dict of methods and parameters
method_parameters = {
        "RGraph":{"gamma":[5,50,200,350,500], "algorithm":["lasso_cd"]}, #use lasso_cd due to convergence issues
        "INNE":{},
        "GMM":{"n_components":range(2,15)},
        "KDE":{},
        "ABOD":{"method":["fast"], "n_neighbors":[60]}, 
        "CBLOF":{"n_clusters":range(2,15), "alpha":[0.7,0.8,0.9], "beta":[3,5,7], "use_weights":[True]},
        "u-CBLOF":{"n_clusters":range(2,15), "alpha":[0.7,0.8,0.9], "beta":[3,5,7], "use_weights":[False]},
        "COF":{"n_neighbors":[5,10,15,20,25,30]},
        "COPOD":{},
        "HBOS":{"n_bins":["auto"]},
        "kNN":{"n_neighbors":range(5,31), "method":["mean"]},
        "kth-NN":{"n_neighbors":range(5,31), "method":["largest"]},
        "IF":{"n_estimators":[1000], "max_samples":[128,256,512,1024]},
        "LMDD":{"n_iter":[100],"dis_measure":["aad"]}, #aad is the same as the MAD
        "LODA":{"n_bins":["auto"]},
        "ensemble-LOF":{"estimators":[[LOF(n_neighbors=k) for k in ensemble_LOF_krange]], "combination_function":[maximization]},
        "LOF":{"n_neighbors":range(5,31)},
        "MCD":{"support_fraction":[0.6,0.7,0.8,0.9], "assume_centered":[True]},
        "OCSVM":{"kernel":["rbf"], "gamma":["auto"], "nu":[0.5,0.6,0.7,0.8,0.9]},
         "PCA":{"n_components":[0.3,0.5,0.7,0.9]}, 
        "SOD":{"n_neighbors":[20, 25 ,30], "ref_set":[10,14,18], "alpha":[0.7,0.8,0.9]},
        "EIF":{"n_estimators":[1000], "max_samples":[128,256,512,1024], "extension_level":[1,2,3]},
        "ODIN":{"n_neighbors":range(5,31)},
        "ECOD":{},
        # "gen2out":
        "AE":{"n_layers":[1,2,3], "shrinkage_factor":[0.2,0.3,0.5], "dropout_rate":[0], "epochs":[200], "validation_size":[0.2], "output_activation":["linear"], "verbose":[0]},
        "VAE":{"n_layers":[1,2,3], "shrinkage_factor":[0.2,0.3,0.5], "dropout_rate":[0], "epochs":[200], "validation_size":[0.2], "output_activation":["linear"], "verbose":[0]},
        "beta-VAE":{"n_layers":[1,2,3], "shrinkage_factor":[0.2,0.3,0.5], "dropout_rate":[0], "epochs":[200], "validation_size":[0.2], "output_activation":["linear"], "gamma":[10,20,50], "verbose":[0]},
        "AnoGAN":{"D_n_layers":[3], "D_shrinkage_factor":[0.3,0.5], "G_n_layers":[3], "G_shrinkage_factor":[0.3,0.5],  "verbose":[0], "epochs":[200]},

        }

#%% 
if method_to_run == "all":
    all_methods_to_run = method_classes
else:
    try:
        all_methods_to_run = {method_to_run:method_classes[method_to_run]}
    except KeyError:
        raise KeyError("Specified method is not found in the list of available methods.")

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

    for method_name, OD_class in all_methods_to_run.items():
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
                
                #use memory efficient COF when too many samples:
                if method_name =="COF" and X.shape[0] > 8000:
                    hyperparameter_setting["method"] = "memory"
                
                #process DeepSVDD differently due to lacking sklearn interface
                #instead: call deepsvdd script from command line with arguments parsed from variables
                if method_name == "DeepSVDD":
                    
                    preprocessed_data_file_name = os.path.join(DeepSVDD_dir, picklefile_name)
                    #preprocess data and write to csv:
                    
                    #check if preprocessed data already exists:, if not preprocess and write data
                    if not os.path.exists(preprocessed_data_file_name):
                        scaler = RobustScaler()
                        
                        X_preprocessed = scaler.fit(X)
                        
                        data_dict = {"X": X_preprocessed, "y": y}
                        
                        pickle.dump(data_dict, open(preprocessed_data_file_name, "wb"))    
                    
                    #make shell call to calculate DeepSVDD
                    s
                    
                    continue #skip rest of loop
                
                
                OD_method = OD_class(**hyperparameter_setting)
                
                #Temporary fix for ECOD:
                if method_name == "ECOD" and hasattr(OD_method, "X_train"):
                    delattr(OD_method, "X_train")
                    
                pipeline = make_pipeline(RobustScaler(), OD_method)
                
                try:
                    pipeline.fit(X)
                except ValueError as e: #Catch error when CBLOF fails due to configuration
                    if str(e) == "Could not form valid cluster separation. Please change n_clusters or change clustering method":
                        print("Separation invalid, skipping this hyperparameter setting")
                        continue
                    else:
                        raise e
                #resolve 
                if method_name in ["AE", "VAE", "beta-VAE"]:
                    
                    gc.collect() 
                    K.clear_session() 
                
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
                
                #write Keras history for relevant neural methods
                if method_name in ["VAE", "beta-VAE", "AE", "AnoGAN"]:
                    if method_name == "AnoGAN":
                        history_df = pd.DataFrame({"discriminator_loss":pipeline[1].hist_loss_discriminator, "generator_loss":pipeline[1].hist_loss_generator})
                    else:
                        history = pipeline[1].history_
                        history_df = pd.DataFrame(history)


                    
                    full_target_dir = os.path.join(log_dir, picklefile_name.replace(".pickle", ""), method_name)
                    target_file_name = os.path.join(log_dir, picklefile_name.replace(".pickle", ""), method_name, hyperparameter_string+".pickle")
                    
                    os.makedirs(full_target_dir, exist_ok=True)
                    with open(target_file_name, 'wb') as handle:
                        pickle.dump(history_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                    full_target_dir = os.path.join(log_dir, picklefile_name.replace(".pickle", ""), method_name)
                    target_file_name = os.path.join(log_dir, picklefile_name.replace(".pickle", ""), method_name, hyperparameter_string+".csv")
                    
                    os.makedirs(full_target_dir, exist_ok=True)

                    history_df.to_csv(target_file_name)

