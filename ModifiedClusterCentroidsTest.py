from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from ModifiedClusterCentroid import ModifiedClusterCentroids

X,y = make_blobs(
    n_samples = 750,
    random_state = 1,
    cluster_std=0.4
)

preproc = ModifiedClusterCentroids()
X_new, y_new = preproc.fit_resample(X,y)
print(X_new)
