import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.base import ClusterMixin
from sklearn.utils.validation import  check_X_y

class ModifiedClusterCentroids(ClusterMixin):
    """
    Zmodyfikowany algorytm ClusterCentroids bazujący na klasteryzacji
    za pomocą algorytmu DBSCAN

    Hiperparametry:
    n_cluster, eps, min_samples, metric, algorithm przyjmują takie same wartości jak w przypadku algorytmu DBSCAN
    CC_strategy -> ('const','auto'): 
    *'const' zmniejsza klasy większościowe automatycznie do klasy mniejszościowej
    *'auto' zmniejsza klasy większościowe na podstawie odchylenia standardowego
    cluster_algorithm->określa jaki algorytm klasteryzacji wykorzystujemy
    min_samples->parametr potrzebny do OPTICS
    max_eps->parametr potrzebny do algorytmu OPTICS określa czy cały obszar będziemy analizować
    """
    def __init__(self, CC_strategy, eps=0.5, metric='euclidean', algorithm='auto', min_samples=5, cluster_algorithm='DBSCAN', max_eps=np.inf):
        self.eps = eps
        self.cluster_algorithm = cluster_algorithm
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.CC_strategy = CC_strategy
        self.max_eps = max_eps

    def rus(self, X, y, n_samples):
        # Wybór losowych próbek z zadanej przestrzeni jako argument funkcji
        X_inc = np.random.choice(len(X), size=n_samples, replace=False)
        return X[X_inc], y[X_inc]

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]

        # Wybór klasy mniejszościowej oraz  klas większościowych
        l, c = np.unique(y, return_counts=True)
        minor_probas = np.amin(c)
        minor_class = l[minor_probas==c]
        major_class = l[minor_probas!=c]
        # Tabela z danymi po zmianie kształtu
        X_resampled = []
        y_resampled = []

        if self.CC_strategy == 'const':
            """ W przypadku hiperparametru 'const'
            redukowana jest liczebność klas większościowych do poziomu 
            Klasy mniejszościowej"""

            # Klasteryzacja DBSCAN lub OPTICS
            if self.cluster_algorithm == 'DBSCAN':
                clustering = DBSCAN(eps=self.eps, metric=self.metric, algorithm=self.algorithm).fit(X[y!=minor_class])
            elif self.cluster_algorithm == 'OPTICS':
                clustering = OPTICS(min_samples=self.min_samples).fit(X[y!=minor_class])
            else:
                raise ValueError('Incorrect cluster_algorithm!')

            # Określenie rozkłądu prawdopodobieństwa apriori pomiędzy klastrami
            l, c = np.unique(clustering.labels_, return_counts=True)

            # Określenie poziomu, do którego bedzie zmniejszana klasa większościowa
            if len(l)==1:
                new_c=int(minor_probas)
                X_selected, y_selected = self.rus(X[y!=minor_class][clustering.labels_==l], y[y!=minor_class][clustering.labels_==l], n_samples=new_c)
                X_resampled.append(X_selected)
                y_resampled.append(y_selected)
                # Dodanie klasy mniejszościowej
                X_resampled.append(X[y==minor_class])
                y_resampled.append(y[y==minor_class])
                X_resampled=np.concatenate(X_resampled)
                y_resampled=np.concatenate(y_resampled)
                return X_resampled, y_resampled

            else:
                # Obliczanie prawdopodobieństwa apriori i deklaracja liczby próbek jaka ma zostać w klastrze
                prob = [i/c.sum() for i in c]
                new_c = [prob[i]*minor_probas for i in range(0, len(c))]
                new_c = np.ceil(new_c)
                
                # Undersampling wewnątrz klastrów
                for label, n_samples in zip(l, new_c):
                    n_samples = int(n_samples)
                    X_selected, y_selected = self.rus(X[y!=minor_class][clustering.labels_==label], y[y!=minor_class][clustering.labels_==label], n_samples=n_samples)
                    X_resampled.append(X_selected)
                    y_resampled.append(y_selected)

                # Dodanie klasy mniejszościowej
                X_resampled.append(X[y==minor_class])
                y_resampled.append(y[y==minor_class])
                X_resampled=np.concatenate(X_resampled)
                y_resampled=np.concatenate(y_resampled)
                return X_resampled, y_resampled

        elif self.CC_strategy == 'auto':
            """ W przypadku hiperparametru 'auto'
            algorytm sam wybiera do jakiego poziomy nastąpi undersampling
            na podstawie odchylenia standardowego w klastrach"""

            # Klasteryzacja 
            if self.cluster_algorithm == 'DBSCAN':
                clustering = DBSCAN(eps=self.eps, metric=self.metric, algorithm=self.algorithm).fit(X[y!=minor_class])
            elif self.cluster_algorithm == 'OPTICS':
                clustering = OPTICS(min_samples=self.min_samples, max_eps=self.max_eps).fit(X[y!=minor_class])
            else:
                raise ValueError('Incorrect cluster_algorithm!')

            # Obliczenia odchylenia standarodowego
            l, c = np.unique(clustering.labels_, return_counts=True)
            # Jesli istnieje jeden klaster  to redukcja nastepuje do wartości klastra mniejszosciowego
            if len(l)==1:
                X_selected, y_selected = self.rus(X[y!=minor_class][clustering.labels_==l], y[y!=minor_class][clustering.labels_==l], n_samples=int(minor_probas))
                X_resampled.append(X_selected)
                y_resampled.append(y_selected)

                # Dodanie klasy mniejszościowej
                X_resampled.append(X[y==minor_class])
                y_resampled.append(y[y==minor_class])
                X_resampled=np.concatenate(X_resampled)
                y_resampled=np.concatenate(y_resampled)
                return X_resampled, y_resampled 

            else:
                std = []
                for i in l:
                    std.append(np.std(X[y!=minor_class][clustering.labels_==i].flatten()))
                std=np.array(std)
                std=std/std.sum()
                # Wybór większej ilości próbek z klastrów o małym odchyleniu tzn. o duzej gestości
                std = [1 - i for i in std]
                std = np.array(std)

                # Mnozenie 1-std/std.sum razy liczbe próbek w klastrze. Im wieksze odchylenie tym mniej próbek zostanie.
                new_c = std*c
                new_c = np.ceil(new_c)
                for label, n_samples in zip(l, new_c):
                    n_samples = int(n_samples)
                    X_selected, y_selected = self.rus(X[y!=minor_class][clustering.labels_==label], y[y!=minor_class][clustering.labels_==label], n_samples=n_samples)
                    X_resampled.append(X_selected)
                    y_resampled.append(y_selected)

                # Dodanie klasy mniejszościowej
                X_resampled.append(X[y==minor_class])
                y_resampled.append(y[y==minor_class])
                X_resampled=np.concatenate(X_resampled)
                y_resampled=np.concatenate(y_resampled)
                return X_resampled, y_resampled 

        else:
            raise ValueError("Incorrect CC_strategy!")         
