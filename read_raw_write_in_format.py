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
import json


target_dir = "formatted_OD_data"

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

dataset_summaries = []

#%% Define filtering function

def get_variance_filter_index(X, max_mode_samples):
    
    _, X_mode_count = mode(X)
    
    variance_filter = np.array([i <= max_mode_samples for i in np.squeeze(X_mode_count)],dtype=bool)
    
    return(variance_filter)
   
    
def preprocess_data(X,y):
    (X_unique, unique_indices) = np.unique(X, axis=0, return_index=True)
    y_unique = y[unique_indices]
    
    n_removed_duplicates = len(y)-len(y_unique)
    print(str(n_removed_duplicates) + " samples were removed due to being duplicates.")
    print("----------------------------------------------------")
    
    variance_filter = get_variance_filter_index(X_unique, train_size_fraction*X_unique.shape[0])
    n_variables_filtered = np.count_nonzero(variance_filter==0)
    X_filtered = X_unique[:,variance_filter]
    for i, v in enumerate(variance_filter):
        if not v:
            print("Variable " + str(i) + " was removed due to having low variance")
            
    data_dict = {"X": X_filtered, "y": y_unique, "n_removed_duplicates": n_removed_duplicates, "n_variables_filtered": n_variables_filtered}
    
    return(data_dict)

def make_dataset_summary(dataset_name, data_dict, categorical_variables, origin):
    
    n_samples, n_variables = data_dict["X"].shape
    n_outliers = int(np.sum(data_dict["y"]))
    outlier_percentage = round(n_outliers/n_samples * 100,2)
    n_removed_duplicates = data_dict["n_removed_duplicates"]
    n_categorical_variables = len(categorical_variables)
    n_numeric_variables = n_variables - n_categorical_variables
    n_variables_filtered = data_dict["n_variables_filtered"]
    
    #summary = pd.DataFrame([[n_samples, n_variables, str(n_outliers) + " (" + str(outlier_percentage) + "%)", n_removed_duplicates, n_numeric_variables, n_categorical_variables, n_variables_filtered]], 
    #                       columns=["#samples", "#variables", "#outliers (%outliers)", "#removed duplicates", "#numeric variables", "#categorical variables", "#removed variables"])

    summary = {"Name": dataset_name,
               "Origin": origin,
               "#samples": n_samples, 
               "#variables": n_variables, 
               "#outliers": n_outliers,
               "%outliers": "("+str(outlier_percentage) + "%)", 
               "#removed duplicates": n_removed_duplicates, 
               "#numeric variables": n_numeric_variables, 
               "#categorical variables": n_categorical_variables, 
               "#removed variables": n_variables_filtered}
    
    return(summary)
#%% ODDS
data_dir = "ODDS_data_raw/matfile_data"
nonmat_data_dir = "ODDS_data_raw/other_data"


matfile_names = os.listdir(data_dir)

HDFlist = ["http.mat", "smtp.mat"] #use MATLAB 7.3 file format (need HDF reader)
black_list = ["ecoli.mat", "breastw.mat", "lympho.mat", "annthyroid.mat"] #ecoli is broken, lympho is removed due to being categorical, breastw has too many outliers %-wise, this is fixed in wbc

train_size_fraction = 1 #can be set to between 0 and 1 in case of cross-validation

with open("ODDS_data_raw/categorical_variables_per_dataset.json", "r") as json_file:
    categorical_variables_per_dataset = json.load(json_file)
    
origin = "ODDS"

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
    
    dataset_name = re.search('(.+?)\.mat', file_name).group(1)
    
    try:
        categorical_variables = categorical_variables_per_dataset[dataset_name]
        print("some categorical variables")
    except KeyError:
        categorical_variables = []
        print("no categorical variables")
    
    data_dict = preprocess_data(X, y)
    
    dataset_summary = make_dataset_summary(dataset_name, data_dict, categorical_variables, origin)
    dataset_summaries.append(dataset_summary)
    
    target_file_name =  dataset_name + ".pickle"
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
    
    dataset_name = re.search('(.+?)\.mat', file_name).group(1)
    
    try:
        categorical_variables = categorical_variables_per_dataset[dataset_name]
        print("some categorical variables")
    except KeyError:
        categorical_variables = []
        print("no categorical variables")
    
    data_dict = preprocess_data(X, y)
    
    dataset_summary = make_dataset_summary(dataset_name, data_dict, categorical_variables, origin)
    dataset_summaries.append(dataset_summary)
    
    target_file_name =  dataset_name + ".pickle"
    target_file_name_with_dir = os.path.join(target_dir, target_file_name)
    pickle.dump(data_dict, open(target_file_name_with_dir, "wb"))    
    
#%%
#arff files
    
#What should be done with the data in order to acquire the proper 11/19 attributes is unknown. I suppose that the first 11 are used, and the nbumps are all omitted.
file_name = "seismic-bumps.arff"
print("----------------------------------------------------")
print("Processing: " + file_name)
print("----------------------------------------------------")
seismic = arff.loadarff(os.path.join(nonmat_data_dir, file_name))

seismic_data = pd.DataFrame(seismic[0])

#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
cat_columns = ["seismic", "seismoacoustic", "shift", "ghazard", "class"]
seismic_data_numerical = pd.get_dummies(seismic_data, prefix_sep="_", columns=cat_columns)

X = seismic_data_numerical.values[:,:-2].astype(np.float64)
y = seismic_data_numerical.values[:,-1].astype(np.float64) #can be made vector as they are complementary

dataset_name = re.search('(.+?)\.arff', file_name).group(1)

try:
    categorical_variables = categorical_variables_per_dataset[dataset_name]
    print("some categorical variables")
except KeyError:
    categorical_variables = []
    print("no categorical variables")

data_dict = preprocess_data(X, y)

dataset_summary = make_dataset_summary(dataset_name, data_dict, categorical_variables, origin)
dataset_summaries.append(dataset_summary)

target_file_name = re.search('(.+?)\.arff', file_name).group(1) + ".pickle"
target_file_name_with_dir = os.path.join(target_dir, target_file_name)
pickle.dump(data_dict, open(target_file_name_with_dir, "wb")) 

#%% Yeast data is commented out due to being undocumented.
# .data/csv files

# file_name = "yeast.data"
# print("----------------------------------------------------")
# print("Processing: " + file_name)
# print("----------------------------------------------------")
# yeast = pd.read_csv(os.path.join(nonmat_data_dir, file_name), delim_whitespace=True, header=None)
# #Cyt

# X = yeast.iloc[:,1:9].values.astype(np.float64)
# y = pd.get_dummies(yeast.iloc[:,9])[["CYT", "NUC", "MIT", "ME3"]].sum(axis=1).values.astype(np.float64)

# dataset_name = re.search('(.+?)\.data', file_name).group(1)

# try:
#     categorical_variables = categorical_variables_per_dataset[dataset_name]
#     print("some categorical variables")
# except KeyError:
#     categorical_variables = []
#     print("no categorical variables")

# data_dict = preprocess_data(X, y)

# dataset_summary = make_dataset_summary(dataset_name, data_dict, categorical_variables)
# dataset_summaries.append(dataset_summary)


# target_file_name = re.search('(.+?)\.data', file_name).group(1) + ".pickle"
# target_file_name_with_dir = os.path.join(target_dir, target_file_name)
# pickle.dump(data_dict, open(target_file_name_with_dir, "wb")) 


#%% Goldstein CSV data


data_dir = "Goldstein_data_raw"
target_dir = "formatted_OD_data"

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

csv_file_names = os.listdir(data_dir)

black_list =  ["annthyroid", "letter", "satellite", "shuttle", "speech", "breast-cancer"]
black_list = [f+"-unsupervised-ad.csv" for f in black_list]
train_size_fraction = 1 #can be set to between 0 and 1 in case of cross-validation

origin="Goldstein"

#%% Write Goldstein CSVs to pickles
for file_name in [f for f in csv_file_names if f not in black_list]:
    
    full_path_filename = os.path.join(data_dir, file_name)
    csv_file = pd.read_csv(full_path_filename)
    print("----------------------------------------------------")
    print("Processing: " + file_name)
    print("----------------------------------------------------")
    X = csv_file.iloc[:,:-1].values.astype(np.float64) 
    y_raw = csv_file.iloc[:,-1]
    y = np.array([1 if v == 'o' else 0 for v in y_raw], dtype=np.float64)
    dataset_name = re.search('(.+?)-unsupervised-ad\.csv', file_name).group(1)
    print(dataset_name)

    categorical_variables = []
    print("no categorical variables")
    
    data_dict = preprocess_data(X, y)
    
    dataset_summary = make_dataset_summary(dataset_name, data_dict, categorical_variables, origin)
    dataset_summaries.append(dataset_summary)
    
    target_file_name =  dataset_name + ".pickle"
    target_file_name_with_dir = os.path.join(target_dir, target_file_name)
    pickle.dump(data_dict, open(target_file_name_with_dir, "wb"))   

#%% GAAL CSV data


data_dir = "GAAL_data_raw"
target_dir = "formatted_OD_data"

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

csv_file_names = os.listdir(data_dir)

black_list = ["Annthyroid", "WDBC"] 
train_size_fraction = 1 #can be set to between 0 and 1 in case of cross-validation

origin = "GAAL"

#%% Write GAAL paper CSVs to pickles:
for file_name in [f for f in csv_file_names if f not in black_list]:
    
    full_path_filename = os.path.join(data_dir, file_name)
    csv_file = pd.read_csv(full_path_filename)
    print("----------------------------------------------------")
    print("Processing: " + file_name)
    print("----------------------------------------------------")
    X = csv_file.iloc[:,2:].values.astype(np.float64) 
    y_raw = csv_file.iloc[:,1]
    y = np.array([0 if v == 'nor' else 1 for v in y_raw], dtype=np.float64)
    dataset_name = file_name.lower()
    print(dataset_name)

    categorical_variables = []
    print("no categorical variables")
    
    data_dict = preprocess_data(X, y)
    
    dataset_summary = make_dataset_summary(dataset_name, data_dict, categorical_variables, origin)
    dataset_summaries.append(dataset_summary)
    
    target_file_name =  dataset_name + ".pickle"
    target_file_name_with_dir = os.path.join(target_dir, target_file_name)
    pickle.dump(data_dict, open(target_file_name_with_dir, "wb"))  




#%% Write ELKI data

data_dir = "ELKI_data_raw"
target_dir = "formatted_OD_data"

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

arff_file_folders = os.listdir(data_dir)

black_list = ["Annthyroid", "Arrhythmia", "Cardiotocography", "HeartDisease", "Pima", "SpamBase"] 
original_file_list = ["Hepatitis_withoutdupl_norm_16.arff", "InternetAds_withoutdupl_norm_19.arff", "PageBlocks_withoutdupl_norm_09.arff", "Parkinson_withoutdupl_norm_75.arff", "Stamps_withoutdupl_norm_09.arff", "Wilt_withoutdupl_norm_05.arff"]

train_size_fraction = 1 #can be set to between 0 and 1 in case of cross-validation

origin = "ELKI"

#%% Write Campos paper ARFF to pickles:
for file_folder, file_name in zip(sorted([f for f in arff_file_folders if f not in black_list]), original_file_list):
    
    full_path_filename = os.path.join(data_dir, file_folder, file_name)
    arff_data = arff.loadarff(full_path_filename)
    print("----------------------------------------------------")
    print("Processing: " + file_name)
    print("----------------------------------------------------")
    arff_df = pd.DataFrame(arff_data[0])
    X = arff_df.iloc[:,:-2].values.astype(np.float64)
    y_raw = arff_df["outlier"]
    y = np.array([0 if v == b'no' else 1 for v in y_raw], dtype=np.float64)
    dataset_name = file_folder.lower()
    print(dataset_name)

    categorical_variables = []
    print("no categorical variables")
    
    data_dict = preprocess_data(X, y)
    
    dataset_summary = make_dataset_summary(dataset_name, data_dict, categorical_variables, origin)
    dataset_summaries.append(dataset_summary)
    
    target_file_name =  dataset_name + ".pickle"
    target_file_name_with_dir = os.path.join(target_dir, target_file_name)
    pickle.dump(data_dict, open(target_file_name_with_dir, "wb"))  

#%% make summary into dataframe and write to latex
summaries_df = pd.DataFrame(dataset_summaries).sort_values("Name")

summaries_df = summaries_df.drop(["#numeric variables", "#categorical variables", "#removed variables"], axis=1,) #remove columns irrelevant to current iteration of research

summaries_df.to_csv("tables/datasets_summaries.csv", index=False)

table_file = open("tables/datasets_table.tex","w")
summaries_df.to_latex(table_file, index=False) 
table_file.close()