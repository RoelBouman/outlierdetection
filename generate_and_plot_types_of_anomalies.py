#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")


#%% peripheral point

peripheral_point_plot = plt.figure()

distribution_1 = np.random.randn(1000,2) + np.array([1,1])

anomalies_1 = np.array([[-3,5], [1, -3.5], [-4,-3]], dtype=np.float32)


sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue", alpha=0.2)
sns.scatterplot(x=anomalies_1[:,0], y=anomalies_1[:,1], marker="X", color="red", s=100)
plt.xlim(-6, 10)
plt.ylim(-6, 10)

plt.xlabel("X$_1$")
plt.ylabel("X$_2$")

#plt.title("Peripheral Anomalies")

plt.savefig("figures/peripheral_point_example.pdf", format="pdf")

#%% enclosed point plot


enclosed_point_plot = plt.figure()

distribution_1 = np.random.randn(1000,2) + np.array([-4,-4])
distribution_2 = np.random.randn(1000,2) + np.array([-4, 4])
distribution_3 = np.random.randn(1000,2) + np.array([4, -4])
distribution_4 = np.random.randn(1000,2) + np.array([4,4])

anomalies_1 = np.array([[0,0], [-0.5,0.3], [0.4,-0.7]], dtype=np.float32)


sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue", alpha=0.2)
sns.scatterplot(x=distribution_2[:,0], y=distribution_2[:,1], color="blue", alpha=0.2)
sns.scatterplot(x=distribution_3[:,0], y=distribution_3[:,1], color="blue", alpha=0.2)
sns.scatterplot(x=distribution_4[:,0], y=distribution_4[:,1], color="blue", alpha=0.2)
sns.scatterplot(x=anomalies_1[:,0], y=anomalies_1[:,1], marker="X", color="red", s=100)
plt.xlim(-8, 8)
plt.ylim(-8, 8)
#plt.legend(loc='upper left')

plt.xlabel("X$_1$")
plt.ylabel("X$_2$")


#plt.title("Enclosed Anomalies")

plt.savefig("figures/enclosed_point_example.pdf", format="pdf")
#%% local_outlier_plot

local_outlier_plot = plt.figure()

distribution_1 = np.random.randn(1000,2) + np.array([1,1])
distribution_2 = np.random.randn(1000,2)/5 + np.array([7,7])


anomalies = np.array([[6.2,6.5], [7.2, 8], [7.9,6.3]])

plt.figure()
sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue", alpha=0.2)
sns.scatterplot(x=distribution_2[:,0], y=distribution_2[:,1], color="blue", alpha=0.2)

sns.scatterplot(x=anomalies[:,0], y=anomalies[:,1], marker="X", color="red", s=100)
plt.xlim(-6, 10)
plt.ylim(-6, 10)
#plt.legend(loc='upper left')

plt.xlabel("X$_1$")
plt.ylabel("X$_2$")


#plt.title("Local Density Anomalies")


plt.savefig("figures/local_outlier_example.pdf", format="pdf")

#%% Global outlier_plot

anomalies = np.array([[8,0], [7.5,1]])

plt.figure()
sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue", alpha=0.2)
sns.scatterplot(x=distribution_2[:,0], y=distribution_2[:,1], color="blue", alpha=0.2)

sns.scatterplot(x=anomalies[:,0], y=anomalies[:,1], marker="X", color="red", s=100)
plt.xlim(-6, 10)
plt.ylim(-6, 10)

plt.xlabel("X$_1$")
plt.ylabel("X$_2$")


#plt.title("Global Density Anomalies")


plt.savefig("figures/global_outlier_example.pdf", format="pdf")
#%% clustered outliers

local_outlier_plot = plt.figure()

distribution_1 = np.random.randn(1000,2) + np.array([1,1])
anomalies_2 = np.random.randn(10,2)/5 + np.array([7,7])

plt.figure()
sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue", alpha=0.2)

sns.scatterplot(x=anomalies_2[:,0], y=anomalies_2[:,1], marker="X", color="red", s=100)
plt.xlim(-6, 10)
plt.ylim(-6, 10)
#plt.legend(loc='upper left')

plt.xlabel("X$_1$")
plt.ylabel("X$_2$")


#plt.title("Clustered Anomalies")

plt.savefig("figures/clustered_outlier_example.pdf", format="pdf")

#%% clustered outliers

local_outlier_plot = plt.figure()

anomalies = np.array([[7,7], [-4,-4], [-4,6]])

plt.figure()
sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue", alpha=0.2)

sns.scatterplot(x=anomalies[:,0], y=anomalies[:,1], marker="X", color="red", s=100)
plt.xlim(-6, 10)
plt.ylim(-6, 10)
#plt.legend(loc='upper left')

plt.xlabel("X$_1$")
plt.ylabel("X$_2$")


#plt.title("Isolated Anomalies")

plt.savefig("figures/isolated_outlier_example.pdf", format="pdf")

#%% univariate outliers

distribution_1 = np.random.multivariate_normal([0,0], [[3,0,],[0,1]], 1000)

anomalies = np.array([[0,6],[8,0]])


plt.figure()
sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue", alpha=0.2)

sns.scatterplot(x=anomalies[:,0], y=anomalies[:,1], marker="X", color="red", s=100)
plt.xlim(-9,9)
plt.ylim(-4,7)
#plt.legend(loc='upper left')

plt.xlabel("X$_1$")
plt.ylabel("X$_2$")

#plt.title("Univariate Anomalies")

plt.savefig("figures/univariate_outlier_example.pdf", format="pdf")

#%% multivariate outliers

distribution_1 = np.random.multivariate_normal([0,0], [[0.5,0],[4,4]], 1000)

anomalies = np.array([[-3,3],[4,-2]])


plt.figure()
sns.scatterplot(x=distribution_1[:,0], y=distribution_1[:,1], color="blue", alpha=0.2)

sns.scatterplot(x=anomalies[:,0], y=anomalies[:,1], marker="X", color="red", s=100)
plt.xlim(-7,7)
plt.ylim(-7,7)
#plt.legend(loc='upper left')

plt.xlabel("X$_1$")
plt.ylabel("X$_2$")

#plt.title("Multivariate Anomalies")

plt.savefig("figures/multivariate_outlier_example.pdf", format="pdf")

