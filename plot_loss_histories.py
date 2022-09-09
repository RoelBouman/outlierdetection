import os
import pickle
import re
import matplotlib.pyplot as plt
import pandas as pd

log_dir = "logs"

method = "DeepSVDD"

datasets = os.listdir(log_dir)

for dataset in datasets:
    method_folder = os.path.join(log_dir,dataset,method)
    
    regex = r"*.pickle"
    
    method_paths = os.listdir(method_folder)
    
    pickle_files = list(filter(lambda x: x.endswith(".csv"), method_paths));

    for file_name in pickle_files:
        log_path = os.path.join(method_folder, file_name)
        #try:
        #history = pickle.load(open(log_path, 'rb'))
        
        history = pd.read_csv(log_path)
        
        history = history.drop(["Unnamed: 0"], axis=1)
        
        plt.figure()
        history.plot()
        plt.title(log_path)

