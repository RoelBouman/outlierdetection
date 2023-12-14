#%% setup
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.utils.utility import precision_n_scores
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import ParameterGrid
from evaluation_metrics import adjusted_precision_n_scores, adjusted_average_precision


formatted_data_dir = "formatted_data"
base_result_dir = "results"
result_dir = "result_dir"
csvresult_dir = "csvresult_dir"
score_dir = "score_dir"

#define score function:
score_functions = {"ROC/AUC": roc_auc_score, 
                   "R_precision": precision_n_scores, 
                   "adjusted_R_precision": adjusted_precision_n_scores, 
                   "average_precision": average_precision_score, 
                   "adjusted_average_precision": adjusted_average_precision}


verbose = True
input_type = "npz"

#%%

def preprocess_detect_outliers(methods, method_parameters, verbose=True, input_type="npz"):
    
    #sort dataset_names based on size: https://stackoverflow.com/questions/20252669/get-files-from-directory-argument-sorting-by-size
    # make a generator for all file paths within dirpath
    all_files = ( os.path.join(basedir, filename) for basedir, dirs, files in os.walk(formatted_data_dir) for filename in files   )
    sorted_files = sorted(all_files, key = os.path.getsize)
    dataset_names = [filename.replace(formatted_data_dir+os.path.sep,"") for filename in sorted_files]
    dataset_names = [dataset_name for dataset_name in dataset_names if dataset_name.endswith(input_type)]
    
    all_methods_to_run = methods
    
    #%% loop over all data, but do not reproduce existing results
    
    target_dir = os.path.join(base_result_dir, result_dir)
    target_csvdir = os.path.join(base_result_dir, csvresult_dir)
    score_csvdir = os.path.join(base_result_dir, score_dir)
    
    if not os.path.exists(score_csvdir):
        os.makedirs(score_csvdir)
        
    for dataset_name in dataset_names:
        
        #print name for reporting purpose
        print("______"+dataset_name+"______")
        
        full_path_filename = os.path.join(formatted_data_dir, dataset_name)
        
        if input_type == "pickle":
            data = pickle.load(open(full_path_filename, 'rb'))
        elif input_type == "npz":
            data = data = np.load(open(full_path_filename, 'rb'))
                        
        X, y = data["X"], np.squeeze(data["y"])
        
        #loop over all methods:
    
        for method_name, OD_class in all_methods_to_run.items():
            print("-" + method_name)
            hyperparameter_grid = method_parameters[method_name]
            hyperparameter_list = list(ParameterGrid(hyperparameter_grid))
            
            #loop over hyperparameter settings
            for hyperparameter_setting in hyperparameter_list:
                
                hyperparameter_string = str(hyperparameter_setting)
                
                if verbose:
                    print(hyperparameter_string)
                
                #check whether results have  been calculated
                full_target_dir = os.path.join(target_dir, dataset_name.replace("."+input_type, ""), method_name)
                target_file_name = os.path.join(target_dir, dataset_name.replace("."+input_type, ""), method_name, hyperparameter_string+".pickle")
                if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
                    if verbose:
                        print(" results already calculated, skipping recalculation")
                else:
                        
                    OD_method = OD_class(**hyperparameter_setting)
                    
                    pipeline = make_pipeline(RobustScaler(), OD_method)
                    
                    pipeline.fit(X)
    
                    outlier_scores = pipeline[1].decision_scores_
                    
                    method_performance = {method_name:{score_name: score_function(y,outlier_scores) for (score_name, score_function) in score_functions.items()}}
                    method_performance_df = pd.DataFrame(method_performance).transpose()
                        
                    os.makedirs(full_target_dir, exist_ok=True)
                    with open(target_file_name, 'wb') as handle:
                        pickle.dump(method_performance_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    #also write csv files for easy manual inspection
                    full_target_csvdir = os.path.join(target_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_csvdir, exist_ok=True)
                    target_csvfile_name = os.path.join(full_target_csvdir, hyperparameter_string+".csv")
                    method_performance_df.to_csv(target_csvfile_name)
                    
                    full_target_scoredir = os.path.join(score_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_scoredir, exist_ok=True)
                    target_scorefile_name = os.path.join(full_target_scoredir, hyperparameter_string+".csv")
                    np.savetxt(target_scorefile_name, outlier_scores)
                    

