#Current procedure does the following:
    #Remove duplicates
    #Filter based on variance

#%% setup
import numpy as np
from scipy.io import loadmat
from scipy.io import arff
from scipy.stats import mode
#from scipy.stats import hypergeom
#from scipy.stats import binom
import h5py
import pandas as pd
import os
import pickle
import re

data_dir = "ODDS_data_raw/matfile_data"
nonmat_data_dir = "ODDS_data_raw/other_data"
target_dir = "formatted_OD_data"

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

matfile_names = os.listdir(data_dir)

HDFlist = ["http.mat", "smtp.mat"] #use MATLAB 7.3 file format (need HDF reader)
black_list = ["ecoli.mat"] #broken file

train_size_fraction = 0.6 #at least 60% of the data is used for training

#%% Define filtering function

def get_variance_filter_index(X, max_mode_samples):
    
    _, X_mode_count = mode(X)
    
    variance_filter = np.array([i <= max_mode_samples for i in np.squeeze(X_mode_count)],dtype=bool)
    
    return(variance_filter)
   
    
def preprocess_data(X,y):
    (X_unique, unique_indices) = np.unique(X, axis=0, return_index=True)
    y_unique = y[unique_indices]
    
    print(str(len(y)-len(y_unique)) + " samples were removed due to being duplicates.")
    print("----------------------------------------------------")
    
    variance_filter = get_variance_filter_index(X_unique, train_size_fraction*X_unique.shape[0])
    X_filtered = X_unique[:,variance_filter]
    for i, v in enumerate(variance_filter):
        if not v:
            print("Variable " + str(i) + " was removed due to having low variance")
            
    data_dict = {"X": X_filtered, "y": y_unique}
    
    return(data_dict)

#%%
# Regular mat files

for file_name in [f for f in matfile_names if f not in HDFlist and f not in black_list]:
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = loadmat(full_path_filename)
    print("----------------------------------------------------")
    print("Processing: " + file_name)
    print("----------------------------------------------------")
    X = mat_file["X"].astype(np.float64) 
    y = mat_file["y"].astype(np.float64)
    
    data_dict = preprocess_data(X, y)
    
    target_file_name = re.search('(.+?)\.mat', file_name).group(1) + ".pickle"
    target_file_name_with_dir = os.path.join(target_dir, target_file_name)
    pickle.dump(data_dict, open(target_file_name_with_dir, "wb"))    
#%%
# HDF mat files
    
for file_name in HDFlist:
    print("----------------------------------------------------")
    print("Processing: " + file_name)
    print("----------------------------------------------------")
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = h5py.File(full_path_filename, 'r')
        
    X = mat_file["X"][()].T.astype(np.float64)
    y = mat_file["y"][()].T.astype(np.float64)
    
    data_dict = preprocess_data(X, y)
    
    target_file_name = re.search('(.+?)\.mat', file_name).group(1) + ".pickle"
    target_file_name_with_dir = os.path.join(target_dir, target_file_name)
    pickle.dump(data_dict, open(target_file_name_with_dir, "wb")) 
    
#%%
#arff files
    
seismic_file_name = "seismic-bumps.arff"
print("----------------------------------------------------")
print("Processing: " + seismic_file_name)
print("----------------------------------------------------")
seismic = arff.loadarff(os.path.join(nonmat_data_dir, seismic_file_name))

seismic_data = pd.DataFrame(seismic[0])

#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
cat_columns = ["seismic", "seismoacoustic", "shift", "ghazard", "class"]
seismic_data_numerical = pd.get_dummies(seismic_data, prefix_sep="_", columns=cat_columns)

X = seismic_data_numerical.values[:,:-2].astype(np.float64)
y = seismic_data_numerical.values[:,-1].astype(np.float64) #can be made vector as they are complementary

data_dict = preprocess_data(X, y)

target_file_name = re.search('(.+?)\.arff', seismic_file_name).group(1) + ".pickle"
target_file_name_with_dir = os.path.join(target_dir, target_file_name)
pickle.dump(data_dict, open(target_file_name_with_dir, "wb")) 

#%%
# .data/csv files

yeast_file_name = "yeast.data"
print("----------------------------------------------------")
print("Processing: " + yeast_file_name)
print("----------------------------------------------------")
yeast = pd.read_csv(os.path.join(nonmat_data_dir, yeast_file_name), delim_whitespace=True, header=None)
#Cyt

X = yeast.iloc[:,1:9].values.astype(np.float64)
y = pd.get_dummies(yeast.iloc[:,9])[["CYT", "NUC", "MIT", "ME3"]].sum(axis=1).values.astype(np.float64)

data_dict = preprocess_data(X, y)

target_file_name = re.search('(.+?)\.data', yeast_file_name).group(1) + ".pickle"
target_file_name_with_dir = os.path.join(target_dir, target_file_name)
pickle.dump(data_dict, open(target_file_name_with_dir, "wb")) 
