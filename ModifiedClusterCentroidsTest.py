from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from ModifiedClusterCentroid import ModifiedClusterCentroids

datasets = 'sonar'
dataset = np.genfromtxt("datasets/%s.csv" % (datasets), delimiter=',')
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

#print(y)
preproc = ModifiedClusterCentroids(CC_strategy='const')
X_new, y_new = preproc.fit_resample(X,y)
y_new = np.reshape(y_new, (X_new.shape[0], 1))

XD = np.concatenate((X_new, y_new), axis=1)



