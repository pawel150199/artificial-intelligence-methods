from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from imblearn.under_sampling import ClusterCentroids
from sklearn.base import clone
from sklearn.datasets import make_blobs
from imbalancelevel import Imbalance

X,y = make_blobs(n_samples=200, n_features=4, random_state=10)

res = ClusterCentroids(random_state=10)

X_res, y_res = res.fit_resample(X, y)
print(res.sampling_strategy_)

