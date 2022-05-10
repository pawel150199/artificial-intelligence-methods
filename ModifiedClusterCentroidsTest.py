from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from ModifiedClusterCentroid import ModifiedClusterCentroids

datasets = 'sonar'
dataset = np.genfromtxt("datasets/%s.csv" % (datasets), delimiter=',')
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
#print(X.shape)
print(y)

preproc = ModifiedClusterCentroids()
X_new, y_new = preproc.fit_resample(X,y)
#print(X_new.shape)
#print(y_new.shape)
