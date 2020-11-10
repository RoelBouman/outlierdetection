import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


result_dir = "D:\\Promotie\\outlier_detection\\result_dir"
figure_dir = "D:\\Promotie\\outlier_detection\\figures"

data_names = os.listdir(result_dir)[:4]

method_names = os.listdir(os.path.join(result_dir, data_names[0]))

k_folds = 5


for data_name in data_names:
    
    mean_precision = np.zeros((len(method_names)))
    range_precision = np.zeros((2,len(method_names)))
    

    for i, method_name in enumerate(method_names):
        
        precision_per_fold = np.zeros((k_folds))
        
        for k in range(k_folds):
            
            
            full_path_filename = os.path.join(result_dir, data_name, method_name, "fold_"+str(k)+"_best_param_scores.pickle")

            partial_result = pickle.load(open(full_path_filename, 'rb'))
      
            precision_per_fold[k] = partial_result["adjusted_average_precision"]["adjusted_average_precision"]
            
            
        
        mean_precision[i] = np.mean(precision_per_fold)
        range_precision[0,i] = mean_precision[i] - np.min(precision_per_fold)
        range_precision[1,i] = np.max(precision_per_fold) - mean_precision[i]
        
    
    plt.figure()
    #plt.bar(range(len(method_names)), mean_precision, yerr = range_precision, tick_label = method_names)
    plt.bar(range(len(method_names)), mean_precision, yerr = range_precision, tick_label = method_names, capsize=6)
    plt.errorbar(range(len(method_names)), mean_precision, fmt="none" )
    plt.title(data_name)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figure_dir,data_name))

