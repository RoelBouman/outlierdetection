import pickle
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare
import scipy.stats
from scikit_posthocs import posthoc_nemenyi_friedman
import math
sns.set()

result_dir = "result_dir"
figure_dir = "figures"

method_blacklist = []
double_dataset_blacklist = ["annthyroid"] #completely cluster together with ODDS datasets
unsolvable_dataset_blacklist = ["speech", "vertebral"]#, "speech_Goldstein"]
own_dataset_blacklist = ["letter-recognition.data"] #own datasets for global/local verification
dataset_blacklist = unsolvable_dataset_blacklist + own_dataset_blacklist# + double_dataset_blacklist 

result_files = os.listdir(result_dir)

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
methods = []
datasets = []
for result_file in result_files:
    full_path_filename = os.path.join(result_dir, result_file)
    
    partial_result = pickle.load(open(full_path_filename, 'rb'))
    
    (data_name, method_name) = re.compile("(.*)_(.*?)_.*").match(result_file).groups()
    
    methods.append(method_name)
    datasets.append(data_name)

methods = list(set(methods))
datasets = list(set(datasets))

#%% Read all metrics from files
metric_dfs = {}
for evaluation_metric in evaluation_metrics:
    
    metric_dfs[evaluation_metric] = pd.DataFrame(index=methods,columns=datasets)

for result_file in result_files:
    full_path_filename = os.path.join(result_dir, result_file)
    
    partial_result = pickle.load(open(full_path_filename, 'rb'))
    
    (data_name, method_name) = re.compile("(.*)_(.*?)_.*").match(result_file).groups()
        
    for evaluation_metric in evaluation_metrics: 
        try: 
            metric_dfs[evaluation_metric][data_name][method_name] = partial_result[evaluation_metric][method_name]
        except KeyError:
            if method_name == "IF":
                metric_dfs[evaluation_metric][data_name][method_name] = partial_result[evaluation_metric]["Isolation Forest"]
        
#%% optional: filter either datasets or methods for which not all methods are in:
    # Also filter blacklisted items.

prune = "methods"        
        
for evaluation_metric in evaluation_metrics:
    metric_dfs[evaluation_metric].drop(method_blacklist, axis=0, inplace=True)
    metric_dfs[evaluation_metric].drop(dataset_blacklist,axis=1,inplace=True)
        
    if prune == "methods":
        metric_dfs[evaluation_metric].dropna(axis=0, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    elif prune == "datasets":
        metric_dfs[evaluation_metric].dropna(axis=1, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    #elif prune == "running":
        #running_dataset = metric_dfs[evaluation_metric].isna().sum().idxmax() 
        #metric_dfs[evaluation_metric].drop(running_dataset, axis=1, inplace=True)
        #metric_dfs[evaluation_metric].dropna(axis=0, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
        

    
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
rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print("iman davenport score: " + str(iman_davenport_score))

print("Critical value: " + str(iman_davenport_critical_value(rank_df)))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_formatted = nemenyi_table.applymap(lambda x: p_value_to_string(x, n_decimals)).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

table_file = open("tables/nemenyi_table_all_datasets.tex","w")
nemenyi_formatted.to_latex(table_file)
table_file.close()

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
plt.show()



#%% clustermap
#Do clustering on percentage of performance, rather than straight AUC

plot_df = metric_dfs["ROC/AUC"].astype(float)

clustermap = sns.clustermap(plot_df.transpose().iloc[:,:], method="average",metric="correlation", figsize=(15,15))

clustermap.savefig("figures/clustermap_all_datasets.eps",format="eps")
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

local_datasets = ["parkinson", "wilt", "aloi", "vowels", "letter", "pen-local", "waveform", "glass", "ionosphere"]


score_df = metric_dfs["ROC/AUC"][local_datasets]
rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print("local:")
print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print ("iman davenport score local: " + str(iman_davenport_score))
print("Critical value: " + str(iman_davenport_critical_value(rank_df)))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_formatted = nemenyi_table.applymap(lambda x: p_value_to_string(x, n_decimals)).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

table_file = open("tables/nemenyi_table_local.tex","w")
nemenyi_formatted.to_latex(table_file)
table_file.close()

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

score_df = metric_dfs["ROC/AUC"]
global_datasets = score_df.columns.difference(local_datasets)
score_df = score_df[global_datasets]
rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print("global:")
print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print ("iman davenport score global: " + str(iman_davenport_score))
print("Critical value: " + str(iman_davenport_critical_value(rank_df)))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_formatted = nemenyi_table.applymap(lambda x: p_value_to_string(x, n_decimals)).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

table_file = open("tables/nemenyi_table_global.tex","w")
nemenyi_formatted.to_latex(table_file)
table_file.close()




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
