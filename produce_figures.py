import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare
import scipy.stats
from scikit_posthocs import posthoc_nemenyi_friedman
sns.set()

prune = "datasets"        

result_dir = "results/csvresult_dir"
figure_dir = "figures"
table_dir = "tables"

os.makedirs(table_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

method_blacklist = []
#TODO: What to do with the large_dataset_blacklist? Currently it is not in sync with the actual paper
large_dataset_blacklist = ["celeba", "backdoor", "fraud"]
double_dataset_blacklist = [] 
unsolvable_dataset_blacklist = ["hrss_anomalous_standard", "wpbc"]
dataset_blacklist = large_dataset_blacklist + unsolvable_dataset_blacklist + double_dataset_blacklist 

rename_datasets = {"hrss_anomalous_optimized":"hrss"}

evaluation_metrics = ["ROC/AUC","R_precision", "adjusted_R_precision", "average_precision", "adjusted_average_precision"]
#%%
def score_to_rank(score_df): #for example score_to_rank(metric_dfs["ROC/AUC"])
    return(score_df.rank(ascending=False).transpose())

def friedman(rank_df):
    return(friedmanchisquare(*[rank_df[col] for col in rank_df.columns]))

def iman_davenport(rank_df): #could also return p-value, but would have to find F value table
    friedman_stat, _ = friedman(rank_df)
    
    N, k = rank_df.shape
    
    iman_davenport_stat = ((N-1)*friedman_stat)/(N*(k-1)-friedman_stat)
    return(iman_davenport_stat)

def iman_davenport_critical_value(rank_df):
    
    N, k = rank_df.shape
        
    return(scipy.stats.f.ppf(0.05, k-1, (k-1)*(N-1)))
        
    

#%%

#First find all datasets and methods used:
datasets = set(os.listdir(result_dir)) - set(dataset_blacklist)
    
methods_per_dataset = []

method_count_per_dataset = {}
max_methods = 0
for dataset in datasets:
    method_folders = os.listdir(os.path.join(result_dir, dataset))
    
    unique_datasets = set(method_folders)-set(method_blacklist)
    
    methods_per_dataset.append(unique_datasets)
    
    method_count_per_dataset[dataset] = len(unique_datasets)
    
    if method_count_per_dataset[dataset] > max_methods:
        max_methods = method_count_per_dataset[dataset]


if prune == "methods":
    methods = set.intersection(*methods_per_dataset)
    
    incomplete_methods = set([x for xs in methods_per_dataset for x in xs]).difference(methods)
    
    if len(incomplete_methods) > 0:
        print("The following methods were not calculated for each dataset:")
        print(incomplete_methods)
    
    methods = list(methods)
elif prune == "datasets":
    methods = set.union(*methods_per_dataset)
    
    datasets = [m  for m in method_count_per_dataset if method_count_per_dataset[m] == max_methods]
    
    incomplete_datasets = list(set(os.listdir(result_dir)) - set(dataset_blacklist) - set(datasets))
    
    if len(incomplete_datasets) > 0:
        print("The following datasets were not calculated for each method:")
        print(incomplete_datasets)



#%% Read all metrics from files

#contains the averaged results
metric_dfs = {}

#contains the full results of all hyperparameters
full_metric_dfs = {}

for evaluation_metric in evaluation_metrics:
    
    metric_dfs[evaluation_metric] = pd.DataFrame(index=methods,columns=datasets)
    full_metric_dfs[evaluation_metric] = pd.DataFrame(index=methods,columns=datasets)

for dataset_name in datasets:
    for method_name in methods:
        
            result_folder_path = os.path.join(result_dir, dataset_name, method_name)
            
            hyperparameter_csvs = os.listdir(result_folder_path)
            hyperparameter_settings = [filename.replace(".csv", "") for filename in hyperparameter_csvs]
            
            results_per_setting = {}
            for hyperparameter_csv, hyperparameter_setting in zip(hyperparameter_csvs, hyperparameter_settings):
                
                full_path_filename = os.path.join(result_folder_path, hyperparameter_csv)
                
                #results_per_setting[hyperparameter_setting] = pickle.load(open(full_path_filename, 'rb'))
                results_per_setting[hyperparameter_setting] = pd.read_csv(full_path_filename)
                
            for evaluation_metric in evaluation_metrics: 
                metric_per_setting = {setting:results[evaluation_metric].values[0] for setting, results in results_per_setting.items()}
                
                average_metric = np.mean(np.fromiter(metric_per_setting.values(), dtype=float))
                metric_dfs[evaluation_metric][dataset_name][method_name] = average_metric
                full_metric_dfs[evaluation_metric][dataset_name][method_name] = metric_per_setting
        
#%% optional: filter either datasets or methods for which not all methods are in:
    # Also filter blacklisted items.


        
for evaluation_metric in evaluation_metrics:
    #metric_dfs[evaluation_metric].drop(method_blacklist, axis=0, inplace=True, errors="ignore")
    #metric_dfs[evaluation_metric].drop(dataset_blacklist,axis=1,inplace=True, errors="ignore")
        
    if prune == "methods":
        metric_dfs[evaluation_metric].dropna(axis=0, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    elif prune == "datasets":
        metric_dfs[evaluation_metric].dropna(axis=1, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    #elif prune == "running":
        #running_dataset = metric_dfs[evaluation_metric].isna().sum().idxmax() 
        #metric_dfs[evaluation_metric].drop(running_dataset, axis=1, inplace=True)
        #metric_dfs[evaluation_metric].dropna(axis=0, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    metric_dfs[evaluation_metric].rename(columns=rename_datasets, inplace=True)

    

#%% see whether datasets are "solvable", and whether they might need to be inverted:
temp_df = metric_dfs["ROC/AUC"]

low_max_datasets= temp_df.columns[temp_df.max() < 0.6]

invertable_datasets = temp_df.columns[np.logical_and(temp_df.max() < 0.6, temp_df.min() < 0.4)]
#list minima:
print("invertable datasets:")
print(invertable_datasets)
print("minima:")
print(temp_df.min().loc[invertable_datasets])
print("maxima:")
print(temp_df.max().loc[invertable_datasets])

unsolvable_datasets = temp_df.columns[np.logical_and(temp_df.max() < 0.6, temp_df.min() >= 0.4)]

print("Unsolvable datasets:")
print(unsolvable_datasets)
print("minima:")
print(temp_df.min().loc[unsolvable_datasets])
print("maxima:")
print(temp_df.max().loc[unsolvable_datasets])
#%% calculate friedman  nemenyi and write to table
#TODO: Calculate Friedman using Tom's exact implementation

#https://stackoverflow.com/questions/6913532/display-a-decimal-in-scientific-notation
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def p_value_to_string(p_value, n_decimals):
    if p_value < 1.0/(10**n_decimals):
        return "<" + format_e(1.0/(10**n_decimals))
    else:
        return str(round(p_value, n_decimals))

#def p_value_marker(val):


#    bold = 'bold' if float(val) < 0.05 else ''


#    return 'font-weight: %s' % bold
n_decimals = 3

score_df = metric_dfs["ROC/AUC"]
n_columns_first_half = int(len(score_df.columns)/2)

header = ["\\rot{"+column+"}" for column in score_df.columns[:n_columns_first_half]]
table_file = open("tables/AUC_all_datasets_first_half.tex","w")
score_df.iloc[:,:n_columns_first_half].astype(float).round(2).to_latex(table_file, header=header, escape=False)
table_file.close()

header = ["\\rot{"+column+"}" for column in score_df.columns[n_columns_first_half:]]
table_file = open("tables/AUC_all_datasets_second_half.tex","w")
score_df.iloc[:,n_columns_first_half:].astype(float).round(2).to_latex(table_file, header=header, escape=False)
table_file.close()


rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print("iman davenport score: " + str(iman_davenport_score))

print("Critical value: " + str(iman_davenport_critical_value(rank_df)))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_table_copy = nemenyi_table.copy(deep=True)
nemenyi_table_copy.columns = ["\\rot{"+column+"}" for column in nemenyi_table_copy.columns] 
nemenyi_formatted = nemenyi_table_copy.applymap(lambda x: p_value_to_string(x, n_decimals)).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

#table_file = open("tables/nemenyi_table_all_datasets.tex","w")
nemenyi_formatted.to_latex("tables/nemenyi_table_all_datasets.tex", hrules=True)
#table_file.close()

#%% Make table summarizing significance and performance results

p_value_threshold = 0.05

result_df = pd.DataFrame()

result_df["Mean Performance"] = score_df.transpose().mean()

result_df["Performance std"] = score_df.transpose().std()

result_df["Performance Range"] = (score_df.transpose().max() - score_df.transpose().min()).astype(float)

method_outperforms = []
for method in result_df.index:
    outperforming_methods = []
    for competing_method in result_df.index:
        if nemenyi_table[method][competing_method] < p_value_threshold and result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
            outperforming_methods.append(competing_method)
    method_outperforms.append(", ".join(outperforming_methods))

result_df["Outperforms"] = method_outperforms

result_df = result_df.sort_values(by="Mean Performance", ascending=False).round(4)

table_file = open("tables/significance_results_all_datasets.tex","w")
result_df.to_latex(table_file)
table_file.close()

#%% plot average percentage of maximum for all datasets

scaled_df = score_df/score_df.max()*100

reordered_index_all = score_df.transpose().mean().sort_values(ascending=False).index

palette = dict(zip(reordered_index_all, sns.color_palette("husl", n_colors=len(reordered_index_all))))

plot_df = (scaled_df).melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index":"method"})
plt.figure()
ax = sns.boxplot(x="method",y="value",data=plot_df, order=reordered_index_all, palette=palette)
ax.set_title("Percentage of maximum performance (ROC/AUC)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("figures/ROCAUC_boxplot_all_datasets.eps",format="eps")
plt.savefig("figures/ROCAUC_boxplot_all_datasets.png",format="png")
plt.savefig("figures/ROCAUC_boxplot_all_datasets.pdf",format="pdf")
plt.show()



#%% clustermap
#Do clustering on percentage of performance, rather than straight AUC

plot_df = metric_dfs["ROC/AUC"].astype(float)

clustermap = sns.clustermap(plot_df.transpose().iloc[:,:], method="average",metric="correlation", figsize=(15,15))

clustermap.savefig("figures/clustermap_all_datasets.eps",format="eps", dpi=1000)
clustermap.savefig("figures/clustermap_all_datasets.png",format="png")
clustermap.savefig("figures/clustermap_all_datasets.pdf",format="pdf")
plt.show()

#%% Make heatmap/table showing significance results at p < 0.05, p < 0.10, p>=0.10
#import matplotlib as mpl

# cmap = sns.color_palette("flare")
# cmap = mpl.cm.viridis
# cmap = mpl.colors.ListedColormap(sns.color_palette("flare").as_hex())
# cmap = mpl.colors.ListedColormap([[1,1,1], [0.4,0,0.4], [0,0,1]]).reversed()
# bounds = [0, 0.05, 0.10, 1]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

# sns.heatmap(nemenyi_table[reordered_index_global].loc[reordered_index_global], cmap = cmap, norm=norm, cbar_kws={"label":"p-value"})
# plt.show()

significance_table = nemenyi_table.astype(str)

for method in nemenyi_table.columns:
    for competing_method in nemenyi_table.columns:
        if nemenyi_table[method].loc[competing_method] < 0.10:
            if nemenyi_table[method].loc[competing_method] < 0.05:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "++"
                else:
                    significance_table.loc[method,competing_method] = "-{}-"
            else:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "+"
                else:
                    significance_table.loc[method,competing_method] = "-"
        else:
            significance_table.loc[method,competing_method] = ""
            
    

significance_table = significance_table[reversed(reordered_index_all)].loc[reordered_index_all]
table_file = open("tables/nemenyi_summary.tex","w")
significance_table.to_latex(table_file)
table_file.close()

significance_table_truncated = significance_table.loc[:, (significance_table == "++").any() | (significance_table == "+").any()]

significance_table_truncated["Mean Performance"] = score_df.transpose().mean().sort_values(ascending=False).round(3)

table_file = open("tables/nemenyi_summary_truncated.tex","w")
column_format = "l" + "c"*(len(significance_table_truncated.columns)-1) +"|r"
header = ["\\rot{"+column+"}" for column in significance_table_truncated.columns[:-1]] + ["\\rot{\\shortstack[l]{\\textbf{Mean}\\\\\\textbf{AUC}}}"]
significance_table_truncated.to_latex(table_file, column_format=column_format, header=header, escape=False)
table_file.close()


#%% Redo nemenyi test and pairwise testing based on the clustering

#%% Local datasets

local_datasets = ["skin", "ionosphere", "glass", "landsat", "fault", "vowels", "pen-local", "letter", "wilt", "nasa", "parkinson", "waveform", "magic.gamma", "pima", "internetads", "speech", "aloi"]#["parkinson", "wilt", "aloi", "vowels", "letter", "pen-local", "glass", "ionosphere", "nasa", "fault", "landsat", "donors"]

#check if all local datasets have been calculated/are not in blacklist:
local_datasets = [dataset for dataset in local_datasets if dataset in metric_dfs["ROC/AUC"].columns]

score_df = metric_dfs["ROC/AUC"][local_datasets]

rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print("local:")
print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print ("iman davenport score local: " + str(iman_davenport_score))
print("Critical value: " + str(iman_davenport_critical_value(rank_df)))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_table_copy = nemenyi_table.copy(deep=True)
nemenyi_table_copy.columns = ["\\rot{"+column+"}" for column in nemenyi_table_copy.columns] 
nemenyi_formatted = nemenyi_table_copy.applymap(lambda x: p_value_to_string(x, n_decimals)).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

#table_file = open("tables/nemenyi_table_local.tex","w")
nemenyi_formatted.to_latex("tables/nemenyi_table_local.tex", hrules=True)
#table_file.close()

#%% Make table summarizing significance and performance results for local datasets

p_value_threshold = 0.05

result_df = pd.DataFrame()

result_df["Mean Performance"] = score_df.transpose().mean()

result_df["Performance std"] = score_df.transpose().std()

result_df["Performance Range"] = (score_df.transpose().max() - score_df.transpose().min()).astype(float)

method_outperforms = []
for method in result_df.index:
    outperforming_methods = []
    for competing_method in result_df.index:
        if nemenyi_table[method][competing_method] < p_value_threshold and result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
            outperforming_methods.append(competing_method)
    method_outperforms.append(", ".join(outperforming_methods))

result_df["Outperforms"] = method_outperforms

result_df = result_df.sort_values(by="Mean Performance", ascending=False).round(4)

table_file = open("tables/significance_results_local.tex","w")
result_df.to_latex(table_file)
table_file.close()

#%% Make boxplot for local datasets
scaled_df = score_df/score_df.max()*100

reordered_index_local = score_df.transpose().mean().sort_values(ascending=False).index



plot_df = (scaled_df).melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index":"method"})
plt.figure()
ax = sns.boxplot(x="method",y="value",data=plot_df, order=reordered_index_local, palette=palette)
ax.set_title("Percentage of maximum performance (ROC/AUC)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("figures/ROCAUC_boxplot_local_datasets.eps",format="eps")
plt.savefig("figures/ROCAUC_boxplot_local_datasets.png",format="png")
plt.savefig("figures/ROCAUC_boxplot_local_datasets.pdf",format="pdf")
plt.show()

#%% Make heatmap/table showing significance results at p < 0.05, p < 0.10, p>=0.10
#import matplotlib as mpl

# cmap = sns.color_palette("flare")
# cmap = mpl.cm.viridis
# cmap = mpl.colors.ListedColormap(sns.color_palette("flare").as_hex())
# cmap = mpl.colors.ListedColormap([[1,1,1], [0.4,0,0.4], [0,0,1]]).reversed()
# bounds = [0, 0.05, 0.10, 1]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

# sns.heatmap(nemenyi_table[reordered_index_global].loc[reordered_index_global], cmap = cmap, norm=norm, cbar_kws={"label":"p-value"})
# plt.show()

significance_table = nemenyi_table.astype(str)

for method in nemenyi_table.columns:
    for competing_method in nemenyi_table.columns:
        if nemenyi_table[method].loc[competing_method] < 0.10:
            if nemenyi_table[method].loc[competing_method] < 0.05:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "++"
                else:
                    significance_table.loc[method,competing_method] = "-{}-"
            else:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "+"
                else:
                    significance_table.loc[method,competing_method] = "-"
        else:
            significance_table.loc[method,competing_method] = ""
            

significance_table = significance_table[reversed(reordered_index_local)].loc[reordered_index_local]
table_file = open("tables/nemenyi_summary_local.tex","w")
significance_table.to_latex(table_file)
table_file.close()

significance_table_truncated = significance_table.loc[:, (significance_table == "++").any() | (significance_table == "+").any()]


significance_table_truncated["Mean Performance"] = score_df.transpose().mean().sort_values(ascending=False).round(3)

table_file = open("tables/nemenyi_summary_local_truncated.tex","w")
column_format = "l" + "c"*(len(significance_table_truncated.columns)-1) +"|r"
header = ["\\rot{"+column+"}" for column in significance_table_truncated.columns[:-1]] + ["\\rot{\\shortstack[l]{\\textbf{Mean}\\\\\\textbf{AUC}}}"]
significance_table_truncated.to_latex(table_file, column_format=column_format, header=header, escape=False)
table_file.close()


#%% Global datasets
non_cluster_datasets = ["vertebral"]
score_df = metric_dfs["ROC/AUC"]
global_datasets = score_df.columns.difference(local_datasets+non_cluster_datasets)
score_df = score_df[global_datasets]

rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print("global:")
print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print ("iman davenport score global: " + str(iman_davenport_score))
print("Critical value: " + str(iman_davenport_critical_value(rank_df)))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_table_copy = nemenyi_table.copy(deep=True)
nemenyi_table_copy.columns = ["\\rot{"+column+"}" for column in nemenyi_table_copy.columns] 
nemenyi_formatted = nemenyi_table_copy.applymap(lambda x: p_value_to_string(x, n_decimals)).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

#table_file = open("tables/nemenyi_table_global.tex","w")
nemenyi_formatted.to_latex("tables/nemenyi_table_global.tex", hrules=True)
#table_file.close()




#%% Make table summarizing significance and performance results for global datasets

p_value_threshold = 0.05

result_df = pd.DataFrame()

result_df["Mean Performance"] = score_df.transpose().mean()

result_df["Performance std"] = score_df.transpose().std()

result_df["Performance Range"] = (score_df.transpose().max() - score_df.transpose().min()).astype(float)

method_outperforms = []
for method in result_df.index:
    outperforming_methods = []
    for competing_method in result_df.index:
        if nemenyi_table[method][competing_method] < p_value_threshold and result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
            outperforming_methods.append(competing_method)
    method_outperforms.append(", ".join(outperforming_methods))

result_df["Outperforms"] = method_outperforms

result_df = result_df.sort_values(by="Mean Performance", ascending=False).round(4)

table_file = open("tables/significance_results_global.tex","w")
result_df.to_latex(table_file)
table_file.close()

#%% Make boxplot for global datasets
scaled_df = score_df/score_df.max()*100

reordered_index_global = score_df.transpose().mean().sort_values(ascending=False).index

#scaled_df = scaled_df.loc[reordered_index]

plot_df = (scaled_df).melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index":"method"})
plt.figure()
ax = sns.boxplot(x="method",y="value",data=plot_df, order=reordered_index_global, palette=palette)
ax.set_title("Percentage of maximum performance (ROC/AUC)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("figures/ROCAUC_boxplot_global_datasets.eps",format="eps")
plt.savefig("figures/ROCAUC_boxplot_global_datasets.png",format="png")
plt.savefig("figures/ROCAUC_boxplot_global_datasets.pdf",format="pdf")

plt.show()

#%% Make heatmap/table showing significance results at p < 0.05, p < 0.10, p>=0.10
#import matplotlib as mpl

# cmap = sns.color_palette("flare")
# cmap = mpl.cm.viridis
# cmap = mpl.colors.ListedColormap(sns.color_palette("flare").as_hex())
# cmap = mpl.colors.ListedColormap([[1,1,1], [0.4,0,0.4], [0,0,1]]).reversed()
# bounds = [0, 0.05, 0.10, 1]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

# sns.heatmap(nemenyi_table[reordered_index_global].loc[reordered_index_global], cmap = cmap, norm=norm, cbar_kws={"label":"p-value"})
# plt.show()

significance_table = nemenyi_table.astype(str)

for method in nemenyi_table.columns:
    for competing_method in nemenyi_table.columns:
        if nemenyi_table[method].loc[competing_method] < 0.10:
            if nemenyi_table[method].loc[competing_method] < 0.05:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "++"
                else:
                    significance_table.loc[method,competing_method] = "-{}-"
            else:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "+"
                else:
                    significance_table.loc[method,competing_method] = "-"
        else:
            significance_table.loc[method,competing_method] = ""
            

significance_table = significance_table[reversed(reordered_index_global)].loc[reordered_index_global]
table_file = open("tables/nemenyi_summary_global.tex","w")
significance_table.to_latex(table_file)
table_file.close()

significance_table_truncated = significance_table.loc[:, (significance_table == "++").any() | (significance_table == "+").any()]

significance_table_truncated["Mean Performance"] = score_df.transpose().mean().sort_values(ascending=False).round(3)

table_file = open("tables/nemenyi_summary_global_truncated.tex","w")
column_format = "l" + "c"*(len(significance_table_truncated.columns)-1) +"|r"
header = ["\\rot{"+column+"}" for column in significance_table_truncated.columns[:-1]] + ["\\rot{\\shortstack[l]{\\textbf{Mean}\\\\\\textbf{AUC}}}"]
significance_table_truncated.to_latex(table_file, column_format=column_format, header=header, escape=False)
table_file.close()
