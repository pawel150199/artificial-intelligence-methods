import numpy as np
from sklearn.cluster import DBSCAN
from imblearn.under_sampling import ClusterCentroids
from sklearn.base import clone, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.preprocessing import StandardScaler

class ModifiedClusterCentroids(ClusterMixin):
    """Modified ClusterCentroids algorithm basen on DBSCAN
    Parameters:
    n_cluster, eps, min_samples, metric, algorithm are the same in DBSCAN
    CC_strategy is in ('const','auto'): 'auto' automaticly make undersampling calculating min form std, 'const' undersamples classes to min probe values from classes, if value is lower than 4 choose 4."""
    def __init__(self, CC_strategy, n_cluster=None, eps=0.7, metric='euclidean', algorithm='auto',):
        self.n_cluster = n_cluster
        self.eps = eps
        self.metric = metric
        self.algorithm = algorithm
        self.CC_strategy = CC_strategy

    def rus(self, X, y, n_samples):
        X_inc = np.random.choice(len(X), size=n_samples, replace=False)
        return X[X_inc], y[X_inc]
    
    def validate_parameters(self):
        if self.CC_strategy not in ['const', 'auto']:
            raise ValueError('CC_strategy incorrect value')

    def fit_resample(self, X, y):
        X = StandardScaler().fit_transform(X)
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]

        #choose majority classes
        l, c = np.unique(y, return_counts=True)
        minor_probas = np.amin(c)
        min_class = l[np.where(minor_probas==c)]

        #Table for resampled dataset
        X_resampled = []
        y_resampled = []

        if self.CC_strategy == 'const':
            #clustering datasets
            clustering = DBSCAN(eps=self.eps, metric=self.metric, algorithm=self.algorithm).fit(X[y[y!=min], :])
            if minor_probas <= 3:
                minor_probas = 10

            l, c = np.unique(clustering.labels_, return_counts=True)
            new_c = np.tile(minor_probas, len(l))

            for label, n_samples in zip(l, new_c):
                n_samples = int(n_samples)
                X_selected, y_selected = self.rus(X[clustering.labels_==label], y[clustering.labels_==label], n_samples=n_samples)
                X_resampled.append(X_selected)
                y_resampled.append(y_selected)
            X_resampled=np.concatenate(X_resampled)
            y_resampled=np.concatenate(y_resampled)
            l_, c_ = np.unique(y_resampled, return_counts=True)
            return X_resampled, y_resampled
