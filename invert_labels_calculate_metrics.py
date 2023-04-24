import pandas as pd
import os
import pickle
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.utils.utility import precision_n_scores
from evaluation_metrics import adjusted_precision_n_scores, adjusted_average_precision


#define score function:
score_functions = {"ROC/AUC": roc_auc_score, 
                   "R_precision": precision_n_scores, 
                   "adjusted_R_precision": adjusted_precision_n_scores, 
                   "average_precision": average_precision_score, 
                   "adjusted_average_precision": adjusted_average_precision}

# In case anomaly detection problems are wrongly defined, labels can be switched in order to recalculate metrics

# inverted datasets:
# Skin originally has 1 being the skin pixel class, and 0 being the noise class. The skin class is however more homogeneous, so labels should be flipped.
# Vertebral consist out of 3 classes, the normal class, and disk hernia/spondilolysthesis. The latter classes are combined and originally defined as 0 in ODDS, but they are conceptually the anomalies.ArithmeticError
# yeast is poorly documented. We've replaced it with yeast6 from EOAD
inverted_datasets = ["skin", "vertebral"]

pickle_dir = "formatted_data"
score_dir = "results/score_dir"
csv_result_dir = "results/csvresult_dir"
result_dir = "results/result_dir"
figure_dir = "figures"
table_dir = "tables"

all_datasets = set(os.listdir(result_dir)) 

for dataset in all_datasets:
    print(dataset)
    
    full_path_filename = os.path.join(pickle_dir, dataset+".pickle")
    
    data = pickle.load(open(full_path_filename, 'rb'))
    X, y = data["X"], np.squeeze(data["y"])
    
    #invert y:
    if dataset in inverted_datasets:
        y_inverted = np.zeros(y.shape)
        y_inverted[y==0] = 1
        y = y_inverted
    
    for method_name in os.listdir(os.path.join(score_dir, dataset)):
        print(method_name)
        score_folder_path = os.path.join(score_dir, dataset, method_name)
        
        hyperparameter_csvs = os.listdir(score_folder_path)
        hyperparameter_settings = [filename.replace(".csv", "") for filename in hyperparameter_csvs]
        
        results_per_setting = {}
        for hyperparameter_csv, hyperparameter_setting in zip(hyperparameter_csvs, hyperparameter_settings):
            print(hyperparameter_csv)
            full_path_filename = os.path.join(score_folder_path, hyperparameter_csv)
            
            outlier_scores = pd.read_csv(full_path_filename, header=None)
            
            method_performance = {method_name:{score_name: score_function(y,outlier_scores) for (score_name, score_function) in score_functions.items()}}
            method_performance_df = pd.DataFrame(method_performance).transpose()
            
            metric_pickle_file = os.path.join(result_dir, dataset, method_name, hyperparameter_csv.replace(".csv", ".pickle"))
            with open(metric_pickle_file, 'wb') as handle:
                pickle.dump(method_performance_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            metric_csv_file = os.path.join(csv_result_dir, dataset, method_name, hyperparameter_csv)

            #also write csv files for easy manual inspection
            method_performance_df.to_csv(metric_csv_file)