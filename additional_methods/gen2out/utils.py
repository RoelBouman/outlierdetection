####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Email	: mengchil@cs.cmu.edu      #
####################################


import numpy as np
from scipy.spatial.distance import cityblock, euclidean

import matplotlib.pyplot as plt


def uni_disk(n, low=0, high=1):
    r = np.random.uniform(low=low, high=high, size=n)  # radius
    theta = np.random.uniform(low=0, high=2*np.pi, size=n)  # angle
    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)
    return x, y

def sythetic_group_anomaly(seed=0):
    np.random.seed(seed)

    x1, y1 = uni_disk(100000)
    x1 *= 5
    y1 *= 5

    x2, y2 = uni_disk(1000)
    x2 = x2 * 1.5 + 10
    y2 = y2 * 1.5 + 5

    x3, y3 = uni_disk(2000)
    x3 = x3 * 6 + 3
    y3 = y3 - 10

    x4 = [11, -2, 13, 14]
    y4 = [0, 9, -10, 10]

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    X_norm = np.array([x, y]).T

    return X_norm

def plot_xray(X, model, idx_arr, line=False):
    plt.scatter(1, 1, s=100, c='k', marker='*')
    xline = 1 / (2 ** np.arange(0, model.min_rate))

    for idx in idx_arr:
        s = model.scores.T[idx].T
        std, mean = np.std(s, axis=1), np.mean(s, axis=1)
        if line:
            plt.plot(xline, mean, c='k', alpha=0.7)
            plt.fill_between(xline, mean-std, mean+std, color='grey', alpha=0.2)

        max_idx = np.argmax(mean)
        plt.scatter(xline[max_idx], mean[max_idx], s=20, c='k')
    
    plt.plot([2 ** (-(model.min_rate - 0.7)), 1.2], [model.threshold, model.threshold], '--', label='Mean + 3 * Std', alpha=0.8, c='r')
    plt.ylim(-0.05, 1.05)
    plt.xlim(2 ** (-(model.min_rate - 0.7)), 1.2)
    plt.xscale('log', base=2)
    plt.xlabel('Qualification Rate', fontsize=20)
    plt.ylabel('Anomaly Score', fontsize=20)
    plt.legend(fontsize=12)

def plot_results(X, model):
    ### Randomly sample when plotting
    idx_arr = np.concatenate([np.arange(300), 
                              np.arange(100000, 100300), 
                              np.arange(101000, 101300), 
                              np.arange(103000, 103004)])

    ### Plot heatmap
    plt.figure(figsize=(4.8, 4))
    plt.hexbin(X[:, 0], X[:, 1], cmap='cool', gridsize=30, bins='log', mincnt=1)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('results/step0_heatmap.png')

    ### Step 1: X-ray plot
    plt.figure(figsize=(4, 4))
    plot_xray(X, model, idx_arr, line=True)
    plt.tight_layout()
    plt.savefig('results/step1_xray_plot.png')

    ### Step 2: Apex extraction
    plt.figure(figsize=(4, 4))
    plot_xray(X, model, idx_arr, line=False)
    plt.tight_layout()
    plt.savefig('results/step2_apex_extraction.png')

    ### Step 3: Outlier grouping 
    c_arr = ['', 'b', 'r', 'y', 'm', 'g', 'c']
    plt.figure(figsize=(4, 4))
    plt.scatter(X[:, 0], X[:, 1], c='lightgrey', alpha=0.5)

    for l in np.unique(model.labels):
        if l != -1:
            idx = np.where(model.labels == l)[0]
            plt.scatter(X[idx, 0], X[idx, 1], c=c_arr[l], label='GA ' + str(l))

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/step3_outlier_grouping.png')

    ### Step 4: Anomaly iso-curves
    man_x, man_y, man_dis = [], [], []
    for i in np.arange(0, model.min_rate, 0.01):
        for j in np.arange(0, 1.01, 0.01):
            ix = 1 / (2 ** i)
            man_x.append(ix)
            man_y.append(j)
            man_dis.append(cityblock([np.log2(ix) / 10, j], [1, 1]))
    man_x, man_y, man_dis = np.array(man_x), np.array(man_y), np.array(man_dis)

    plt.figure(figsize=(4.8, 4))
    plt.scatter(man_x, man_y, c=man_dis, cmap='gist_rainbow', alpha=0.1)
    plt.colorbar()
    plt.scatter(1, 1, s=100, c='k', marker='*')

    xline = 1 / (2 ** np.arange(0, model.min_rate))
    for idx in idx_arr:
        if model.labels[idx] != -1:
            c = c_arr[model.labels[idx]]
            s = model.scores.T[idx].T
            std, mean = np.std(s, axis=1), np.mean(s, axis=1)
            plt.plot(xline, mean, c=c, alpha=0.05)
            max_idx = np.argmax(mean)
            plt.scatter(xline[max_idx], mean[max_idx], s=20, c=c)
    for l in np.unique(model.labels):
        if l != -1:
            plt.plot([], [], '-o', c=c_arr[l], label='GA ' + str(l))

    plt.xscale('log', base=2)
    plt.ylim(-0.05, 1.05)
    plt.xlim(2 ** (-(model.min_rate - 0.7)), 1.2)
    plt.xlabel('Qualification Rate', fontsize=20)
    plt.ylabel('Anomaly Score', fontsize=20)
    plt.legend(fontsize=12, loc=4)
    plt.tight_layout()
    plt.savefig('results/step4_anomaly_isocurves.png')

    ### Step 5: Scoring
    plt.figure(figsize=(4.4, 4))

    for idx, s in enumerate(model.s_arr):
        ymin, ymax = np.min(s), np.max(s)
        Q1, Q3 = np.percentile(s, 25), np.percentile(s, 75)
        m = np.median(s)
        plt.scatter([idx, idx], [ymin, ymax], facecolors='none', edgecolors='lightgrey')
        plt.plot([idx, idx], [Q1, Q3], c='grey', linewidth=0.9)
        plt.plot([idx-0.12, idx+0.12], [Q1, Q1], c='grey', linewidth=0.9)
        plt.plot([idx-0.12, idx+0.12], [Q3, Q3], c='grey', linewidth=0.9)
        plt.plot([idx-0.24, idx+0.24], [m, m], c='r', linewidth=3)

    plt.xticks(np.arange(len(model.s_arr)), ['GA '+str(i+1) for i in range(len(model.s_arr))], fontsize=12)
    plt.xlabel('Generalized Anomaly ID', fontsize=20)
    plt.ylabel('Distribution of\nAnomaly Score', fontsize=20)
    plt.tight_layout()
    plt.savefig('results/step5_scoring.png')