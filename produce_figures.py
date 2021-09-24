import pickle
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import math
sns.set()

result_dir = "result_dir"
figure_dir = "figures"

method_blacklist = []
double_dataset_blacklist = ["letter_Goldstein", "annthyroid_Goldstein","wbc_Goldstein"] #completely cluster together with ODDS datasets
unsolvable_dataset_blacklist = ["speech", "vertebral", "speech_Goldstein"]
dataset_blacklist = double_dataset_blacklist + unsolvable_dataset_blacklist

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

#%%
metric_dfs = {}
for evaluation_metric in evaluation_metrics:
    
    metric_dfs[evaluation_metric] = pd.DataFrame(index=methods,columns=datasets)

for result_file in result_files:
    full_path_filename = os.path.join(result_dir, result_file)
    
    partial_result = pickle.load(open(full_path_filename, 'rb'))
    
    (data_name, method_name) = re.compile("(.*)_(.*?)_.*").match(result_file).groups()
        
    for evaluation_metric in evaluation_metrics: 
    
        metric_dfs[evaluation_metric][data_name][method_name] = partial_result[evaluation_metric][method_name]
        
        
#%% optional: filter either datasets or methods for which not all methods are in:
    # Also filter blacklisted items.

prune = "running"        
        
for evaluation_metric in evaluation_metrics:
    metric_dfs[evaluation_metric].drop(method_blacklist, axis=0, inplace=True)
    metric_dfs[evaluation_metric].drop(dataset_blacklist,axis=1,inplace=True)
        
    if prune == "methods":
        metric_dfs[evaluation_metric].dropna(axis=0, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    elif prune == "datasets":
        metric_dfs[evaluation_metric].dropna(axis=1, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    elif prune == "running":
        running_dataset = metric_dfs[evaluation_metric].isna().sum().idxmax()
        metric_dfs[evaluation_metric].drop(running_dataset, axis=1, inplace=True)
        metric_dfs[evaluation_metric].dropna(axis=0, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
        


#%% boxplots for 
for evaluation_metric in evaluation_metrics:
    
    plot_df = metric_dfs[evaluation_metric].melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index":"method"})
    plt.figure()
    ax = sns.boxplot(x="method",y="value",data=plot_df)
    ax.set_title(evaluation_metric)
    plt.xticks(rotation=45)
    plt.show()
    
    
#%% calculate friedman  nemenyi and write to table
#TODO: Calculate Friedman using Tom's exact implementation

# def round_to_string(cell):
#     if cell < 0.05:
#         return("\\textbf{"+str(round(cell,2))+"}")
#     else:
#         return(str(round(cell,2)))
    
    
    

#def p_value_marker(val):


#    bold = 'bold' if float(val) < 0.05 else ''


#    return 'font-weight: %s' % bold


score_df = metric_dfs["ROC/AUC"]
rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print ("iman davenport score: " + str(iman_davenport_score))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_formatted = nemenyi_table.round(2).applymap(str).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

table_file = open("tables/nemenyi_table.tex","w")
nemenyi_formatted.to_latex(table_file)
table_file.close()


#%% plot average rank
average_rank = rank_df.mean()

plt.figure()
average_rank.plot.bar()
plt.show()

#%% plot average percentage of maximum

average_perc_max = (score_df/score_df.max()).mean(axis=1)

plt.figure()
average_perc_max.plot.bar()
plt.show()



plot_df = (score_df/score_df.max()*100).melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index":"method"})
plt.figure()
ax = sns.boxplot(x="method",y="value",data=plot_df)
ax.set_title("Percentage of maximum performance (ROC/AUC)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/ROCAUC_boxplot.eps",format="eps")
plt.show()

perc_of_max = (score_df/score_df.max()*100).transpose()

#%% Plot performance for data set size

#%% Perform hierarchical clustering based on correlation
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

# plot_df = metric_dfs["ROC/AUC"]

# Q = plt.figure()
# Z = linkage(plot_df, method='average', metric="correlation", optimal_ordering=True,)
# P = dendrogram(Z, labels=plot_df.index, leaf_rotation=45, color_threshold=max(Z[:,2]))
# ordering = P['ivl']
#%% Plot correlations between method results

#sns.pairplot(plot_df.transpose()[ordering]).set(xlim=[0,1]).set(ylim=[0,1])

#correlation_matrix = plot_df.transpose().astype(float).corr()

#%% combine plots

plot_df = metric_dfs["ROC/AUC"].astype(float)

def dendrogram_pairplot(score_df):
    n_methods = len(score_df.index)
    
    height = int(math.ceil(n_methods * 1.3))
    dendrogram_height = height-n_methods
    
    f = plt.figure(figsize=(n_methods*2,n_methods*2), )
    
    gs = f.add_gridspec(height,n_methods)
    
    dendrogram_ax = f.add_subplot(gs[0:(dendrogram_height), 0:])
    
    Z = linkage(plot_df, method='average', metric="correlation", optimal_ordering=True,)
    P = dendrogram(Z, labels=plot_df.index, leaf_rotation=45, color_threshold=max(Z[:,2]))
    labels = P['ivl']
    plt.xticks(ticks=[])
    
    
    for i in range(n_methods):
        for j in range(n_methods):
            ax = f.add_subplot(gs[dendrogram_height+i,j])
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set(xticks=[0, 1])
            ax.set(yticks=[0, 1])
            if i == j:
                pass           
            else:
                ax.plot(score_df.loc[labels[j]],score_df.loc[labels[i]], 'b.')
                
            if i == n_methods-1:
                ax.set_xlabel(labels[j])     
                ax.xaxis.label.set_size(20)
            else:
                ax.set(xticks=[])
                
            if j == 0:
                ax.set_ylabel(labels[i])
                ax.yaxis.label.set_size(20)
            else:
                ax.set(yticks=[])
    gs.tight_layout(f)
    f.autofmt_xdate()
    
dendrogram_pairplot(plot_df)
plt.tight_layout()
plt.savefig("figures/pairplot.eps",format="eps")
    

#%% clustermap

clustermap = sns.clustermap(plot_df.transpose(), method="average",metric="correlation")

clustermap.savefig("figures/biclustering.eps",format="eps")


#%% biclustering


from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering


model = SpectralBiclustering(n_clusters=4)
model.fit(plot_df.transpose())

fit_data = plot_df.transpose().values
fit_data = fit_data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data)
plt.show()