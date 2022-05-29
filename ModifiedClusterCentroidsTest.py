from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from ModifiedClusterCentroid import ModifiedClusterCentroids
import matplotlib.pyplot as plt 

# Pobieranie danych
#datasets = 'sonar'
#dataset = np.genfromtxt("datasets/%s.csv" % (datasets), delimiter=',')
#X = dataset[:, :-1]
#y = dataset[:, -1].astype(int)

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights= [0.2, 0.8]
)
#print(X)
#print(y)
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='DBSCAN')
X_DBSCAN, y_DBSCAN= preproc.fit_resample(X,y)
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='OPTICS')
X_OPTICS, y_OPTICS= preproc.fit_resample(X,y)

# Przed udersamplingiem DBSCAN
fig, ax = plt.subplots(2,2, figsize=(15,7))
ax[0,0].scatter(*X.T, c=y)
ax[0,0].set_xlim(-4,4)
ax[0,0].set_ylim(-4,4)
ax[0,0].set_xlabel('Feature 0')
ax[0,0].set_ylabel('Feature 1')
ax[0,0].set_title('Before Undersampling - DBSCAN')
# Po udersamplingu DBSCAN
ax[0,1].scatter(*X_DBSCAN.T, c=y_DBSCAN)
ax[0,1].set_xlim(-4,4)
ax[0,1].set_ylim(-4,4)
ax[0,1].set_xlabel('Feature 0')
ax[0,1].set_ylabel('Feature 1')
ax[0,1].set_title('After Undersampling - DBSCAN')
# Przed udersamplingiem OPTICS
ax[1,0].scatter(*X.T, c=y)
ax[1,0].set_xlim(-4,4)
ax[1,0].set_ylim(-4,4)
ax[1,0].set_xlabel('Feature 0')
ax[1,0].set_ylabel('Feature 1')
ax[1,0].set_title('Before Undersampling - OPTICS')
# Po udersamplingu OPTICS
ax[1,1].scatter(*X_OPTICS.T, c=y_OPTICS)
ax[1,1].set_xlim(-4,4)
ax[1,1].set_ylim(-4,4)
ax[1,1].set_xlabel('Feature 0')
ax[1,1].set_ylabel('Feature 1')
ax[1,1].set_title('After Undersampling - OPTICS')

plt.tight_layout()
plt.show()
#y_new = np.reshape(y_new, (X_new.shape[0], 1))

#XD = np.concatenate((X_new, y_new), axis=1)



