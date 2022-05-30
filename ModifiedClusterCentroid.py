import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.base import ClusterMixin
from sklearn.utils.validation import  check_X_y
from sklearn.preprocessing import StandardScaler

class ModifiedClusterCentroids(ClusterMixin):
    """
    Zmodyfikowany algorytm ClusterCentroids bazujący na klasteryzacji
    za pomocą algorytmu DBSCAN

    Parametry:
    n_cluster, eps, min_samples, metric, algorithm przyjmują takie same wartości jak w przypadku algorytmu DBSCAN
    CC_strategy -> ('const','auto'): 
    *'const' zmniejsza klasy większościowe automatycznie do klasy mniejszościowej
    *'auto' zmniejsza klasy większościowe na podstawie odchylenia standardowego
    cluster_algorithm->określa jaki algorytm klasteryzacji wykorzystujemy
    min_samples->parametr potrzebny do OPTICS
    """
    def __init__(self, CC_strategy, eps=0.7, metric='euclidean', algorithm='auto', min_samples=5, cluster_algorithm='DBSCAN'):
        self.eps = eps
        self.cluster_algorithm = cluster_algorithm
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.CC_strategy = CC_strategy

    def rus(self, X, y, n_samples):
        X_inc = np.random.choice(len(X), size=n_samples, replace=False)
        return X[X_inc], y[X_inc]

    def fit_resample(self, X, y):
        X = StandardScaler().fit_transform(X)
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]

        # Wybór klas większościowej
        l, c = np.unique(y, return_counts=True)
        minor_probas = np.amin(c)
        major_class = l[minor_probas!=c]
        print(major_class)

        # Tabela z danymi po zmianie kształtu
        X_resampled = []
        y_resampled = []

        if self.CC_strategy == 'const':
            # Klasteryzacja DBSCAN lub OPTICS
            if self.cluster_algorithm == 'DBSCAN':
                clustering = DBSCAN(eps=self.eps, metric=self.metric, algorithm=self.algorithm).fit(X[y==major_class])
            elif self.cluster_algorithm == 'OPTICS':
                clustering = OPTICS(min_samples=self.min_samples).fit(X[y==major_class])
            else:
                raise ValueError('Incorrect cluster_algorithm!')
            
            #if minor_probas <= 3:
                #minor_probas = 10

            l, c = np.unique(clustering.labels_, return_counts=True)
            print(l)
            print(clustering.labels_.shape)
            print(y[major_class==y])
            prob = [i/len(y[major_class==y]) for i in c]
            print(prob)
            new_c = [prob[i]*minor_probas for i in range(0, len(c))]
            new_c = np.round(new_c)
            print(new_c)
            for label, n_samples in zip(l, new_c):
                n_samples = int(n_samples)
                #print(X[np.where(clustering.labels_==label)])
                X_selected, y_selected = self.rus(X[y==major_class][clustering.labels_==label], y[y==major_class][clustering.labels_==label], n_samples=n_samples)
                X_resampled.append(X_selected)
                y_resampled.append(y_selected)
            X_resampled.append(X[y!=major_class])
            y_resampled.append(y[y!=major_class])
            X_resampled=np.concatenate(X_resampled)
            y_resampled=np.concatenate(y_resampled)
            l_, c_ = np.unique(y_resampled, return_counts=True)
            return X_resampled, y_resampled

        elif self.CC_strategy == 'auto':
            # Klasteryzacja
            if self.cluster_algorithm == 'DBSCAN':
                clustering = DBSCAN(eps=self.eps, metric=self.metric, algorithm=self.algorithm).fit(X)
            elif self.cluster_algorithm == 'OPTICS':
                clustering = OPTICS(min_samples=self.min_samples).fit(X)
            else:
                raise ValueError('Incorrect cluster_algorithm!')

            #if minor_probas <= 3:
                #minor_probas = 10

            l, c = np.unique(clustering.labels_, return_counts=True)
            std = []
            for i in l:
                print(X[clustering.labels_==i])
                std.append(np.std(X[clustering.labels_==i].flatten()))
            std=np.array(std)
            std=std/std.sum()
            print(l,c)
            print('Odchylenie standardowe: \n', std)
        
            #testowanko
            new_c = np.floor(std*c)
            new_c = np.round(new_c)
            print(new_c)
            for label, n_samples in zip(l, new_c):
                n_samples = int(n_samples)
                X_selected, y_selected = self.rus(X[clustering.labels_==label], y[clustering.labels_==label], n_samples=n_samples)
                X_resampled.append(X_selected)
                y_resampled.append(y_selected)
            X_resampled=np.concatenate(X_resampled)
            y_resampled=np.concatenate(y_resampled)
            l_, c_ = np.unique(y_resampled, return_counts=True)
            return X_resampled, y_resampled 

        else:
            raise ValueError("Incorrect CC_strategy!")         
