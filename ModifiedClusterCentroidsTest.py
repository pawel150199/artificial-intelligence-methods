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
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights= [0.2, 0.8]
)
#print(X)
#print(y)
preproc = ModifiedClusterCentroids(CC_strategy='const')
X_new, y_new = preproc.fit_resample(X,y)

fig, ax = plt.subplots(1,2, figsize=(15,7))
ax[0].scatter(*X.T, c=y)
ax[0].set_xlim(-4,4)
ax[0].set_ylim(-4,4)
ax[0].set_xlabel('Feature 0')
ax[0].set_ylabel('Feature 1')
ax[0].set_title('Before Undersampling')
ax[1].scatter(*X_new.T, c=y_new)
ax[1].set_xlim(-4,4)
ax[1].set_ylim(-4,4)
ax[1].set_xlabel('Feature 0')
ax[1].set_ylabel('Feature 1')
ax[1].set_title('After Undersampling')

plt.show()
#y_new = np.reshape(y_new, (X_new.shape[0], 1))

#XD = np.concatenate((X_new, y_new), axis=1)



