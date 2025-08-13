from Gait_Week1 import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

class SupervisedPCAMI:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = None
        self.selected_indices = None

    def fit(self, X, Y):
        """
        X: shape (samples, features)
        Y: shape (samples, targets)
        """
        self.pca = PCA(n_components=min(X.shape[1], 100))
        X_pca = self.pca.fit_transform(X)

        mi_scores = []
        for i in range(X_pca.shape[1]):
            mi = 0
            for j in range(Y.shape[1]):
                mi += mutual_info_regression(X_pca[:, [i]], Y[:, j])[0]
            mi_scores.append(mi / Y.shape[1])

        self.selected_indices = np.argsort(mi_scores)[::-1][:self.n_components]

    def transform(self, X):
        X_pca = self.pca.transform(X)
        return X_pca[:, self.selected_indices]

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X)

insole_l = gait_insole.insole_l
scaler = StandardScaler()
insole_l = scaler.fit_transform(insole_l)  # insole_r shape: (4920, 1024)
print(insole_l.shape)

pca = PCA(n_components=5)
X_reduced = pca.fit_transform(insole_l)     # 4920,50
print(X_reduced.shape)

explained = pca.explained_variance_ratio_  # shape: (50,)
cumulative = np.cumsum(explained)
print(f"Cumulative Variance: {cumulative[-1]*100:.2f}%")

# Visualization
plt.plot(cumulative)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(X_reduced[:, 0], label='PC1')
plt.show()