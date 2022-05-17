from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs, make_classification
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def rus(X, y, n_samples):
    X_inc = np.random.choice(len(X), size=n_samples, replace=False)
    return X[X_inc], y[X_inc]

X,y = make_blobs(
    n_samples = 750,
    random_state = 1,
    cluster_std=0.4
)

X = StandardScaler().fit_transform(X)
clustering = DBSCAN(eps=0.3).fit(X)

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].scatter(X[:,0], X[:,1])
ax[0].set_xlabel('feature 0')
ax[0].set_ylabel('feature 1')
ax[0].set_title('Class')
x = clustering.labels_
l, c = np.unique(x, return_counts=True)
new_c = c/2
X_ = []
y_ = []
for label, n_samples in zip(l, new_c):
    n_samples = int(n_samples)
    X_selected, y_selected = rus(X[x==label], y[x==label], n_samples=n_samples)
    X_.append(X_selected)
    y_.append(y_selected)
X_=np.concatenate(X_)
y_=np.concatenate(y_)
l_, c_ = np.unique(y_, return_counts=True)
print(l)
print(c)
print(l_)
print(c_)
print(x)
ax[1].scatter(X[:,0], X[:,1], c=x)
ax[1].set_xlabel('feature 0')
ax[1].set_ylabel('feature 1')
ax[1].set_title('Class')
plt.tight_layout()
plt.show()
