#%% setup
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from timeit import default_timer as timer

pickle_dir = "formatted_OD_data"
result_dir = "speed_result_dir"

#picklefile_names = os.listdir(pickle_dir) #this can be adjusted to only include smaller datasets.
picklefile_names = ["pima","wbc","ionosphere","vowels","thyroid","cardio","letter", "seismic-bumps"]
picklefile_names = [name+".pickle" for name in picklefile_names]

#%% Define parameter settings and methods

from pyod.models.cof import COF

random_state = 1457969831 #generated using np.random.randint(0, 2**31 -1)


#nested dict of methods and parameters
methods = {
        "COF_mem":COF(n_neighbors=20)
        }

#%% loop over all data, but do not reproduce existing results



#%% 

#make pickle file directory
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
            
        pipeline = make_pipeline(RobustScaler(), OD_method)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        start = timer()
        OD_method.fit(X)
        end = timer()
        exec_time = end - start
        print(exec_time)
        
        
        method_performance = {picklefile_name.replace(".pickle",""):{method_name:exec_time}}
        method_performance_df = pd.DataFrame(method_performance).transpose()
        target_csvfile_name = os.path.join(target_dir, picklefile_name.replace(".pickle", "_"+method_name+"_results.csv"))
        method_performance_df.to_csv(target_csvfile_name)
            
