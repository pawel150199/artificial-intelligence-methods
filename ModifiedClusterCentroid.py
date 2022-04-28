import numpy as np
from sklearn.cluster import DBSCAN
from imblearn.under_sampling import ClusterCentroids
from sklearn.base import BaseUnderSampler
from sklearn.base import clone

class ModifiedClusterCentroids(BaseUnderSampler):
    def ClusterCentroids()