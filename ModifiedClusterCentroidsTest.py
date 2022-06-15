from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from ModifiedClusterCentroid import ModifiedClusterCentroids
import matplotlib.pyplot as plt 

"""
Kod słuzy do przetestowania autorskiego algorytmu i wizualizacji wyników
Do tego wykorzystywany jest synetetyczny zbiór danych
"""
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    weights= [0.8, 0.2]
)
# Pobieranie danych
#datasets = 'sonar'
#dataset = np.genfromtxt("datasets/%s.csv" % (datasets), delimiter=',')
#X = dataset[:, :-1]
#y = dataset[:, -1].astype(int)

# Undersampling zbioru danych
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='DBSCAN')
X_auto, y_auto= preproc.fit_resample(X,y)
preproc = ModifiedClusterCentroids(CC_strategy='auto', cluster_algorithm='DBSCAN')
X_const, y_const= preproc.fit_resample(X,y)
preproc = ModifiedClusterCentroids(CC_strategy='auto', cluster_algorithm='OPTICS')
X_OPTICS_a, y_OPTICS_a= preproc.fit_resample(X,y)
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='OPTICS')
X_OPTICS_c, y_OPTICS_c= preproc.fit_resample(X,y)

# Wyświetlenie wizualizacji
fig, ax = plt.subplots(2,3, figsize=(15,7))
# Przed udersamplingiem DBSCAN
ax[0,0].scatter(*X.T, c=y)
ax[0,0].set_xlim(-4,4)
ax[0,0].set_ylim(-4,4)
ax[0,0].set_xlabel('Feature 0')
ax[0,0].set_ylabel('Feature 1')
ax[0,0].set_title('Before Undersampling - DBSCAN')
# Po udersamplingu DBSCAN - auto
ax[0,1].scatter(*X_auto.T, c=y_auto)
ax[0,1].set_xlim(-4,4)
ax[0,1].set_ylim(-4,4)
ax[0,1].set_xlabel('Feature 0')
ax[0,1].set_ylabel('Feature 1')
ax[0,1].set_title('After Undersampling - DBSCAN-auto')
# Przed udersamplingiem DBSCAN - const
ax[1,0].scatter(*X_const.T, c=y_const)
ax[1,0].set_xlim(-4,4)
ax[1,0].set_ylim(-4,4)
ax[1,0].set_xlabel('Feature 0')
ax[1,0].set_ylabel('Feature 1')
ax[1,0].set_title('After undersampling - DBSCAN-const')
# Po udersamplingu OPTICS - auto
ax[1,1].scatter(*X_OPTICS_a.T, c=y_OPTICS_a)
ax[1,1].set_xlim(-4,4)
ax[1,1].set_ylim(-4,4)
ax[1,1].set_xlabel('Feature 0')
ax[1,1].set_ylabel('Feature 1')
ax[1,1].set_title('After Undersampling - OPTICS-auto')
# Po udersamplingu OPTICS - const
ax[0,2].scatter(*X_OPTICS_c.T, c=y_OPTICS_c)
ax[0,2].set_xlim(-4,4)
ax[0,2].set_ylim(-4,4)
ax[0,2].set_xlabel('Feature 0')
ax[0,2].set_ylabel('Feature 1')
ax[0,2].set_title('After Undersampling - OPTICS-const')

plt.tight_layout()
#plt.show()
plt.savefig('Results/comparision_own_algorithms.png')
#y_new = np.reshape(y_new, (X_new.shape[0], 1))

#XD = np.concatenate((X_new, y_new), axis=1)



