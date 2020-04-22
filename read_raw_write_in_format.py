#%% setup
import numpy as np
from scipy.io import loadmat
from scipy.io import arff
import h5py
import pandas as pd
import os
import pickle
import re

data_dir = "D:\\Promotie\\ODDS_data_raw\\matfile_data"
nonmat_data_dir = "D:\\Promotie\\ODDS_data_raw\\other_data"
target_dir = "D:\\Promotie\\formatted_OD_data"

matfile_names = os.listdir(data_dir)

HDFlist = ["http.mat", "smtp.mat"] #use MATLAB 7.3 file format (need HDF reader)
black_list = ["ecoli.mat"] #broken file
#%%
# Regular mat files

for file_name in [f for f in matfile_names if f not in HDFlist and f not in black_list]:
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = loadmat(full_path_filename)
    
    data_dict = {"X": mat_file["X"], "y": mat_file["y"]}
    
    target_file_name = re.search('(.+?)\.mat', file_name).group(1) + ".pickle"
    target_file_name_with_dir = os.path.join(target_dir, target_file_name)
    pickle.dump(data_dict, open(target_file_name_with_dir, "wb"))    
#%%
# HDF mat files
    
for file_name in HDFlist:
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = h5py.File(full_path_filename, 'r')
        
    data_dict = {"X": mat_file["X"].value.T, "y": mat_file["y"].value.T}
    
    target_file_name = re.search('(.+?)\.mat', file_name).group(1) + ".pickle"
    target_file_name_with_dir = os.path.join(target_dir, target_file_name)
    pickle.dump(data_dict, open(target_file_name_with_dir, "wb")) 
    
#%%
#arff files
    
seismic_file_name = "seismic-bumps.arff"
seismic= arff.loadarff(os.path.join(nonmat_data_dir, seismic_file_name))

seismic_data = pd.DataFrame(seismic[0])

#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
cat_columns = ["seismic", "seismoacoustic", "shift", "ghazard", "class"]
seismic_data_numerical = pd.get_dummies(seismic_data, prefix_sep="_", columns=cat_columns)

X = seismic_data_numerical.values[:,:-2]
y = seismic_data_numerical.values[:,-1] #can be made vector as they are complementary

data_dict = {"X": X, "y": y}

target_file_name = re.search('(.+?)\.arff', seismic_file_name).group(1) + ".pickle"
target_file_name_with_dir = os.path.join(target_dir, target_file_name)
pickle.dump(data_dict, open(target_file_name_with_dir, "wb")) 

#%%
# .data/csv files

yeast_file_name = "yeast.data"
yeast = pd.read_csv(os.path.join(nonmat_data_dir, yeast_file_name), delim_whitespace=True, header=None)
#Cyt

X = yeast.iloc[:,1:9].values
y = pd.get_dummies(yeast.iloc[:,9])[["CYT", "NUC", "MIT", "ME3"]].sum(axis=1).values


data_dict = {"X": X, "y": y}

target_file_name = re.search('(.+?)\.data', yeast_file_name).group(1) + ".pickle"
target_file_name_with_dir = os.path.join(target_dir, target_file_name)
pickle.dump(data_dict, open(target_file_name_with_dir, "wb")) 