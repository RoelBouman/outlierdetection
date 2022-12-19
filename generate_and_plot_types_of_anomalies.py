#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")


#%% peripheral point

peripheral_point_plot = plt.figure()

distribution_1 = np.random.randn(1000,2) + np.array([1,1])

anomalies_1 = np.array([[-3,5], [1, -3.5], [-4,-3]], dtype=np.float32)


sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], label="Cluster 1", color="blue", alpha=0.2)
sns.scatterplot(x=anomalies_1[:,0], y=anomalies_1[:,1], marker="X", label="Outlier", color="red")
plt.xlim(-6, 10)
plt.ylim(-6, 10)
#plt.legend(loc='upper left')

plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("figures/peripheral_point_example.eps", format="eps")

#%% enclosed point plot


enclosed_point_plot = plt.figure()

distribution_1 = np.random.randn(1000,2) + np.array([-4,-4])
distribution_2 = np.random.randn(1000,2) + np.array([-4, 4])
distribution_3 = np.random.randn(1000,2) + np.array([4, -4])
distribution_4 = np.random.randn(1000,2) + np.array([4,4])

anomalies_1 = np.array([[0,0], [-0.5,0.3], [0.4,-0.7]], dtype=np.float32)


sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue")
sns.scatterplot(x=distribution_2[:,0], y=distribution_2[:,1], color="blue")
sns.scatterplot(x=distribution_3[:,0], y=distribution_3[:,1], color="blue")
sns.scatterplot(x=distribution_4[:,0], y=distribution_4[:,1], color="blue")
sns.scatterplot(x=anomalies_1[:,0], y=anomalies_1[:,1], marker="X", color="red")
plt.xlim(-8, 8)
plt.ylim(-8, 8)
#plt.legend(loc='upper left')

plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("figures/peripheral_point_example.eps", format="eps")
#%% local_outlier_plot

local_outlier_plot = plt.figure()

distribution_1 = np.random.randn(1000,2) + np.array([1,1])
distribution_2 = np.random.randn(500,2)/5 + np.array([7,7])


anomalies_1 = np.array([[-3,5], [1, -3.5], [-4,-3]], dtype=np.float32)
anomalies_2 = np.array([[6.2,6.5], [7.2, 8], [7.9,6.3]])

anomalies = np.concatenate((anomalies_1, anomalies_2))

plt.figure()
sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue", alpha=0.2, label="Cluster 1")
sns.scatterplot(x=distribution_2[:,0], y=distribution_2[:,1], color="green", alpha=0.2, label="Cluster 2")

sns.scatterplot(x=anomalies[:,0], y=anomalies[:,1], marker="X", color="red", label="outlier")
plt.xlim(-6, 10)
plt.ylim(-6, 10)
#plt.legend(loc='upper left')

plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("figures/local_outlier_example.eps", format="eps")

#%% clustered outliers

local_outlier_plot = plt.figure()

distribution_1 = np.random.randn(1000,2) + np.array([1,1])
anomalies_2 = np.random.randn(10,2)/5 + np.array([7,7])

plt.figure()
sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue")

sns.scatterplot(x=anomalies_2[:,0], y=anomalies_2[:,1], marker="X", color="red",  label="Outlier")
plt.xlim(-6, 10)
plt.ylim(-6, 10)
#plt.legend(loc='upper left')

plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("figures/local_outlier_example.eps", format="eps")