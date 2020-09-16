import pickle
import os
import numpy as np
import pandas as pd

from pyod.models.knn import KNN 

result_dir = "D:\\Promotie\\outlier_detection\\result_dir"

data_names = os.listdir(result_dir)

method_names = [x.replace(".pickle", "") for x in os.listdir(os.path.join(result_dir, data_names[0]))]


for data_name in data_names:
    

    for method_name in method_names:
        full_path_filename = os.path.join(result_dir, data_name, method_name, )
        partial_result = pickle.load(open(full_path_filename, 'rb'))
    

