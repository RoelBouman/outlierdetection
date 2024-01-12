#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")


#%% peripheral point

# Peripheral point plot
distribution_1 = np.random.randn(1000, 2) + np.array([1, 1])
anomalies_1 = np.array([[-3, 5], [1, -3.5], [-4, -3]], dtype=np.float32)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.scatterplot(x=distribution_1[:, 0], y=distribution_1[:, 1], color="blue", alpha=0.2, ax=axes[0])
sns.scatterplot(x=anomalies_1[:, 0], y=anomalies_1[:, 1], marker="X", color="red", s=100, ax=axes[0])
axes[0].set_xlim(-6, 10)
axes[0].set_ylim(-6, 10)
axes[0].set_xlabel("X$_1$")
axes[0].set_ylabel("X$_2$")
axes[0].set_title("Peripheral Anomalies", fontsize=20)

# Enclosed point plot
distribution_1 = np.random.randn(1000, 2) + np.array([-4, -4])
distribution_2 = np.random.randn(1000, 2) + np.array([-4, 4])
distribution_3 = np.random.randn(1000, 2) + np.array([4, -4])
distribution_4 = np.random.randn(1000, 2) + np.array([4, 4])

anomalies_1 = np.array([[0, 0], [-0.5, 0.3], [0.4, -0.7]], dtype=np.float32)

sns.scatterplot(x=distribution_1[:, 0], y=distribution_1[:, 1], color="blue", alpha=0.2, ax=axes[1])
sns.scatterplot(x=distribution_2[:, 0], y=distribution_2[:, 1], color="blue", alpha=0.2, ax=axes[1])
sns.scatterplot(x=distribution_3[:, 0], y=distribution_3[:, 1], color="blue", alpha=0.2, ax=axes[1])
sns.scatterplot(x=distribution_4[:, 0], y=distribution_4[:, 1], color="blue", alpha=0.2, ax=axes[1])
sns.scatterplot(x=anomalies_1[:, 0], y=anomalies_1[:, 1], marker="X", color="red", s=100, ax=axes[1])
axes[1].set_xlim(-8, 8)
axes[1].set_ylim(-8, 8)
axes[1].set_xlabel("X$_1$")
axes[1].set_ylabel("X$_2$")
axes[1].set_title("Enclosed Anomalies", fontsize=20)

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plot


fig.savefig("figures/enclosed-peripheral_point_example.pdf", format="pdf")

plt.show()
#%% local_outlier_plot

# Local outlier plot
local_outlier_plot = plt.figure()

distribution_1 = np.random.randn(1000, 2) + np.array([1, 1])
distribution_2 = np.random.randn(1000, 2) / 5 + np.array([7, 7])

anomalies = np.array([[6.2, 6.5], [7.2, 8], [7.9, 6.3]])

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.scatterplot(x=distribution_1[:, 0], y=distribution_1[:, 1], color="blue", alpha=0.2, ax=axes[0])
sns.scatterplot(x=distribution_2[:, 0], y=distribution_2[:, 1], color="blue", alpha=0.2, ax=axes[0])
sns.scatterplot(x=anomalies[:, 0], y=anomalies[:, 1], marker="X", color="red", s=100, ax=axes[0])
axes[0].set_xlim(-6, 10)
axes[0].set_ylim(-6, 10)
axes[0].set_xlabel("X$_1$")
axes[0].set_ylabel("X$_2$")
axes[0].set_title("Local Density Anomalies", fontsize=20)

# Global outlier plot
anomalies = np.array([[8, 0], [7.5, 1]])

sns.scatterplot(x=distribution_1[:, 0], y=distribution_1[:, 1], color="blue", alpha=0.2, ax=axes[1])
sns.scatterplot(x=distribution_2[:, 0], y=distribution_2[:, 1], color="blue", alpha=0.2, ax=axes[1])
sns.scatterplot(x=anomalies[:, 0], y=anomalies[:, 1], marker="X", color="red", s=100, ax=axes[1])
axes[1].set_xlim(-6, 10)
axes[1].set_ylim(-6, 10)
axes[1].set_xlabel("X$_1$")
axes[1].set_ylabel("X$_2$")
axes[1].set_title("Global Density Anomalies", fontsize=20)

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plot


fig.savefig("figures/global-local_outlier_example.pdf", format="pdf")

plt.show()
#%% clustered outliers

# Clustered outliers plot
local_outlier_plot = plt.figure()

distribution_1 = np.random.randn(1000, 2) + np.array([1, 1])
anomalies_2 = np.random.randn(10, 2) / 5 + np.array([7, 7])

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.scatterplot(x=distribution_1[:, 0], y=distribution_1[:, 1], color="blue", alpha=0.2, ax=axes[0])
sns.scatterplot(x=anomalies_2[:, 0], y=anomalies_2[:, 1], marker="X", color="red", s=100, ax=axes[0])
axes[0].set_xlim(-6, 10)
axes[0].set_ylim(-6, 10)
axes[0].set_xlabel("X$_1$")
axes[0].set_ylabel("X$_2$")
axes[0].set_title("Clustered Anomalies", fontsize=20)

# Isolated outliers plot
anomalies = np.array([[7, 7], [-4, -4], [-4, 6]])

sns.scatterplot(x=distribution_1[:, 0], y=distribution_1[:, 1], color="blue", alpha=0.2, ax=axes[1])
sns.scatterplot(x=anomalies[:, 0], y=anomalies[:, 1], marker="X", color="red", s=100, ax=axes[1])
axes[1].set_xlim(-6, 10)
axes[1].set_ylim(-6, 10)
axes[1].set_xlabel("X$_1$")
axes[1].set_ylabel("X$_2$")
axes[1].set_title("Isolated Anomalies", fontsize=20)

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plot


fig.savefig("figures/isolated-clustered_outlier_example.pdf", format="pdf")
plt.show()
#%% univariate outliers

# Univariate outliers plot
distribution_1 = np.random.multivariate_normal([0, 0], [[3, 0], [0, 1]], 1000)
anomalies = np.array([[0, 6], [8, 0]])

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.scatterplot(x=distribution_1[:, 0], y=distribution_1[:, 1], color="blue", alpha=0.2, ax=axes[0])
sns.scatterplot(x=anomalies[:, 0], y=anomalies[:, 1], marker="X", color="red", s=100, ax=axes[0])
axes[0].set_xlim(-9, 9)
axes[0].set_ylim(-4, 7)
axes[0].set_xlabel("X$_1$")
axes[0].set_ylabel("X$_2$")
axes[0].set_title("Univariate Anomalies", fontsize=20)

# Multivariate outliers plot
distribution_1 = np.random.multivariate_normal([0, 0], [[0.5, 0], [4, 4]], 1000)
anomalies = np.array([[-3, 3], [4, -2]])

sns.scatterplot(x=distribution_1[:, 0], y=distribution_1[:, 1], color="blue", alpha=0.2, ax=axes[1])
sns.scatterplot(x=anomalies[:, 0], y=anomalies[:, 1], marker="X", color="red", s=100, ax=axes[1])
axes[1].set_xlim(-7, 7)
axes[1].set_ylim(-7, 7)
axes[1].set_xlabel("X$_1$")
axes[1].set_ylabel("X$_2$")
axes[1].set_title("Multivariate Anomalies", fontsize=20)

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plot


fig.savefig("figures/multivariate-univariate_outlier_example.pdf", format="pdf")
plt.show()
