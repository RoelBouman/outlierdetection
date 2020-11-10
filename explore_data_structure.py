#%% setup
import numpy as np
from scipy.io import loadmat
from scipy.io import arff
import h5py
import pandas as pd
import os

data_dir = "D:\Promotie\ODDS_data_raw\matfile_data"
nonmat_data_dir = "D:\Promotie\ODDS_data_raw\other_data"

matfile_names = os.listdir(data_dir)

HDFlist = ["http.mat", "smtp.mat"] #use MATLAB 7.3 file format (need HDF reader)

#%% Check if all files adhere to the same naming conventions
#- two different filetypes exist
for file_name in [f for f in matfile_names if f not in HDFlist]:
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = loadmat(full_path_filename)
    print(full_path_filename)
    print(mat_file.keys())
    
    
for file_name in HDFlist:
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = h5py.File(full_path_filename, 'r')
    print(full_path_filename)
    print(list(mat_file.keys()))
    
#%% Check all shapes of data:
#HDF files are transposed wrt shape
for file_name in [f for f in matfile_names if f not in HDFlist]:
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = loadmat(full_path_filename)
    
    print("X shape: ", mat_file["X"].shape)
    print("X type: ", mat_file["X"].dtype)
    print("y shape: ", mat_file["y"].shape)
    print("y type: ", mat_file["y"].dtype)
    
    
for file_name in HDFlist:
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = h5py.File(full_path_filename, 'r')
    
    print("X shape: ", mat_file["X"].shape)
    print("X type: ", mat_file["X"].dtype)
    print("y shape: ", mat_file["y"].shape)
    print("y type: ", mat_file["y"].dtype)
    
#%% Check for missing values
# ecoli.mat heeft 2352 missing values
for file_name in [f for f in matfile_names if f not in HDFlist]:
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = loadmat(full_path_filename)
    print(full_path_filename)
    print("X NaN: ", np.sum(np.isnan(mat_file["X"])))
    print("y NaN: ", np.sum(np.isnan(mat_file["y"])))
    
    
for file_name in HDFlist:
    full_path_filename = os.path.join(data_dir, file_name)
    mat_file = h5py.File(full_path_filename, 'r')
    print(full_path_filename)
    print("X NaN: ", np.sum(np.isnan(mat_file["X"])))
    print("y NaN: ", np.sum(np.isnan(mat_file["y"])))
    
#%% number of missings in ecoli.mat per column
#Currently all elements of ecoli[X] are loading as NaNs
ecoli = loadmat(os.path.join(data_dir, "ecoli.mat"), chars_as_strings=False)

X = ecoli["X"]
for i in range(X.shape[1]):
    print("col "+str(i)+" NaN: ",np.sum(np.isnan(X[:,i])))
    
#%% manually verify arff file:
    
seismic = arff.loadarff(os.path.join(nonmat_data_dir, "seismic-bumps.arff"))

seismic_data = pd.DataFrame(seismic[0])

#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
cat_columns = ["seismic", "seismoacoustic", "shift", "ghazard", "class"]
seismic_data_numerical = pd.get_dummies(seismic_data, prefix_sep="_", columns=cat_columns)
    
X = seismic_data_numerical.values[:,:-2]
y = seismic_data_numerical.values[:,-1] #can be made vector as they are complementary

for i in range(X.shape[1]):
    print("col "+str(i)+" NaN: ",np.sum(np.isnan(X[:,i])))
    
#%% manually verify .data file:
# Uses delim_whitespace option  because file is unstructured in separating whitespace
#10  different classes, which are used for outliers?
yeast = pd.read_csv(os.path.join(nonmat_data_dir, "yeast.data"), delim_whitespace=True, header=None)

X = yeast.iloc[:,1:9].values
y = pd.get_dummies(yeast.iloc[:,9])

for i in range(X.shape[1]):
    print("col "+str(i)+" NaN: ",np.sum(np.isnan(X[:,i])))