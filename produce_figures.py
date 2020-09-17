import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


result_dir = "D:\\Promotie\\outlier_detection\\result_dir"
figure_dir = "D:\\Promotie\\outlier_detection\\figures"

data_names = os.listdir(result_dir)

method_names = [x.replace(".pickle", "") for x in os.listdir(os.path.join(result_dir, data_names[0]))]


for data_name in data_names[:6]:
    
    mean_precision = np.zeros((len(method_names)))
    range_precision = np.zeros((2,len(method_names)))
    quantiles_precision = np.zeros((2,len(method_names)))

    for i, method_name in enumerate(method_names):
        full_path_filename = os.path.join(result_dir, data_name, method_name+ ".pickle")

        partial_result = pickle.load(open(full_path_filename, 'rb'))
        
        mean_precision[i] = np.mean(partial_result["mean_test_score"])
        range_precision[0,i] = mean_precision[i]-np.min(partial_result["mean_test_score"])
        range_precision[1,i] = np.max(partial_result["mean_test_score"])-mean_precision[i]
        quantiles_precision[0,i] = mean_precision[i]-np.quantile(partial_result["mean_test_score"], 0.025)
        quantiles_precision[1,i] = np.quantile(partial_result["mean_test_score"], 0.975)-mean_precision[i]
    
    plt.figure()
    #plt.bar(range(len(method_names)), mean_precision, yerr = range_precision, tick_label = method_names)
    plt.bar(range(len(method_names)), mean_precision, yerr = quantiles_precision, tick_label = method_names, capsize=6)
    plt.errorbar(range(len(method_names)), mean_precision, kwargs)
    plt.title(data_name)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figure_dir,data_name))

