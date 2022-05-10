import numpy as np
from sklearn.cluster import DBSCAN
from imblearn.under_sampling import ClusterCentroids
from sklearn.base import clone, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

class ModifiedClusterCentroids(ClusterMixin):
    
    def __init__(self, n_cluster=None, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto',):
        self.n_cluster = n_cluster
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm

    def rus(self, X, y, n_samples):
        X_inc = np.random.choice(len(X), size=n_samples, replace=False)
        return X[X_inc], y[X_inc]
    
    def fit(self, X, y):
        """Find the classes statistics before to perform sampling"""
        pass

    def fit_resample(self, X, y):
        indices = []
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        X_resampled, y_resampled = [], []
        
        #learning apriori probability of classes
        classes_disribution = []
        for i in self.classes_:
                n = 0
                for j in y:
                    if j == i:
                        n += 1
                classes_disribution.append(n)

        #clustering datasets
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, algorithm=self.algorithm).fit(X)
        l, c = np.unique(clustering.labels_, return_counts=True)
        print(l, "\n\n\n", c)
        new_c = c/2
        for label, n_samples in zip(l, new_c):
            n_samples = int(n_samples)
            X_selected, y_selected = self.rus(X[clustering.labels_==label], y[clustering.labels_==label], n_samples=n_samples)
            X_resampled.append(X_selected)
            y_resampled.append(y_selected)
        X_resampled=np.concatenate(X_resampled)
        y_resampled=np.concatenate(y_resampled)
        l_, c_ = np.unique(y_selected, return_counts=True)
        return X_resampled, y_resampled
            
        
        
        
            
        
            
