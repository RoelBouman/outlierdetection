#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

#%%
plt.figure()

#%%

plt.figure()

distribution_1 = np.random.randn(1000,2) + np.array([1,1])

outliers_1 = np.array([[-3,5], [1, -3.5], [-4,-3]], dtype=np.float32)


sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], label="Cluster 1", color="blue")
sns.scatterplot(x=outliers_1[:,0], y=outliers_1[:,1], marker="X", label="Outlier", color="red")
plt.xlim(-6, 10)
plt.ylim(-6, 10)
plt.legend(loc='upper left')

plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("figures/global_outlier_example.eps", format="eps")

#%%

plt.figure()

distribution_2 = np.random.randn(500,2)/5 + np.array([7,7])

outliers_2 = np.array([[6.2,6.5], [7.2, 8], [7.9,6.3]])

outliers = np.concatenate((outliers_1, outliers_2))

plt.figure()
sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], label="Cluster 1", color="blue")
sns.scatterplot(x=distribution_2[:,0], y=distribution_2[:,1], label="Cluster 2", color="green")

sns.scatterplot(x=outliers[:,0], y=outliers[:,1], marker="X", label="Outlier", color="red")
plt.xlim(-6, 10)
plt.ylim(-6, 10)
plt.legend(loc='upper left')

plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("figures/local_outlier_example.eps", format="eps")

#%%