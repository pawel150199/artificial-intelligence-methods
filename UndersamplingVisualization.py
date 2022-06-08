from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from ModifiedClusterCentroid import ModifiedClusterCentroids
import matplotlib.pyplot as plt 
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss

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
#MCC - DBSCAN - const
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='DBSCAN')
X_DBSCAN, y_DBSCAN= preproc.fit_resample(X,y)
#MCC - OPTICS - const
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='OPTICS')
X_OPTICS, y_OPTICS= preproc.fit_resample(X,y)
#MCC - DBSCAN - auto
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='DBSCAN')
X_DBSCAN_auto, y_DBSCAN_auto= preproc.fit_resample(X,y)
#MCC - OPTICS - auto
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='OPTICS')
X_OPTICS_auto, y_OPTICS_auto= preproc.fit_resample(X,y)
#RUS
preproc = RandomUnderSampler(random_state=1234)
X_RUS, y_RUS = preproc.fit_resample(X,y)
#CC
preprocs = ClusterCentroids(random_state=1234)
X_CC, y_CC = preproc.fit_resample(X,y)
#NearMiss
preprocs = NearMiss(version=1)
X_NM, y_NM = preproc.fit_resample(X,y)



# Przed udersamplingiem DBSCAN-const
fig, ax = plt.subplots(2,4, figsize=(15,7))
ax[0,0].scatter(*X.T, c=y)
ax[0,0].set_xlim(-5,5)
ax[0,0].set_ylim(-5,5)
ax[0,0].set_xlabel('Feature 0')
ax[0,0].set_ylabel('Feature 1')
ax[0,0].set_title('Before Undersampling' )
# Po udersamplingu DBSCAN-const
ax[0,1].scatter(*X_DBSCAN.T, c=y_DBSCAN)
ax[0,1].set_xlim(-5,5)
ax[0,1].set_ylim(-5,5)
ax[0,1].set_xlabel('Feature 0')
ax[0,1].set_ylabel('Feature 1')
ax[0,1].set_title('After Undersampling - DBSCAN - const')
# Po udersamplingu OPTICS
ax[1,0].scatter(*X_OPTICS.T, c=y_OPTICS)
ax[1,0].set_xlim(-5,5)
ax[1,0].set_ylim(-5,5)
ax[1,0].set_xlabel('Feature 0')
ax[1,0].set_ylabel('Feature 1')
ax[1,0].set_title('After Undersampling - OPTICS - const')
# Po udersamplingu RUS
ax[1,1].scatter(*X_RUS.T, c=y_RUS)
ax[1,1].set_xlim(-5,5)
ax[1,1].set_ylim(-5,5)
ax[1,1].set_xlabel('Feature 0')
ax[1,1].set_ylabel('Feature 1')
ax[1,1].set_title('After Undersampling - RUS')
# Po udersamplingu CC
ax[0,2].scatter(*X_CC.T, c=y_CC)
ax[0,2].set_xlim(-5,5)
ax[0,2].set_ylim(-5,5)
ax[0,2].set_xlabel('Feature 0')
ax[0,2].set_ylabel('Feature 1')
ax[0,2].set_title('After Undersampling - CC')
# Po udersamplingu NearMiss
ax[1,2].scatter(*X_NM.T, c=y_NM)
ax[1,2].set_xlim(-5,5)
ax[1,2].set_ylim(-5,5)
ax[1,2].set_xlabel('Feature 0')
ax[1,2].set_ylabel('Feature 1')
ax[1,2].set_title('After Undersampling - NearMiss')
# Przed udersamplingiem DBSCAN-const
ax[0,3].scatter(*X_OPTICS_auto.T, c=y_OPTICS_auto)
ax[0,3].set_xlim(-5,5)
ax[0,3].set_ylim(-5,5)
ax[0,3].set_xlabel('Feature 0')
ax[0,3].set_ylabel('Feature 1')
ax[0,3].set_title('After Undersampling - OPTICS - auto')
# Po udersamplingu DBSCAN-const
ax[1,3].scatter(*X_DBSCAN_auto.T, c=y_DBSCAN_auto)
ax[1,3].set_xlim(-5,5)
ax[1,3].set_ylim(-5,5)
ax[1,3].set_xlabel('Feature 0')
ax[1,3].set_ylabel('Feature 1')
ax[1,3].set_title('After Undersampling - DBSCAN - auto')

plt.tight_layout()
plt.show()
plt.savefig("Results/visualization.png", dpi=200)
#y_new = np.reshape(y_new, (X_new.shape[0], 1))




