import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from ModifiedClusterCentroid import ModifiedClusterCentroids
from Addons.ImbalanceLevel import Imbalance

"""
Przeprowadzono doświadczenie w celu porównania jak
radzą sobie równe algorytmy oversampligu
w porównaniu z własnym algortytmem
"""

# Klasyfikatory
clfs = {
    'GNB': GaussianNB(),
    'SVC': SVC(),
    'kNN': KNeighborsClassifier(),
    'Linear SVC': LinearSVC(random_state=1234, tol=1e-5)
}

# Metody undersampligu
preprocs = {
    'none': None,
    'RUS' : RandomUnderSampler(random_state=123),
    'CC': ClusterCentroids(random_state=1234),
    'MCC': ModifiedClusterCentroids(CC_strategy='const')
}

if __name__=='__main__':
    # Walidacja krzyzowa
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1234)

    # Tablice z wynikami
    scores = np.zeros((len(preprocs), n_splits*n_repeats, len(clfs)))

    # Zbiór danych
    datasets = 'yeast6'
    dataset = np.genfromtxt("datasets/%s.csv" % (datasets) , delimiter=',')
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    # Przeprowadzenie eksperymentu
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for preproc_id, preproc_name in enumerate(preprocs):
            if preprocs[preproc_name] == None:
                X_res, y_res = X[train], y[train]
            else:
                X_res, y_res = preprocs[preproc_name].fit_resample(X[train],y[train])

            for clf_id, clf_name in enumerate(clfs):
                clf = clfs[clf_name]
                clf.fit(X_res, y_res)
                y_pred = clf.predict(X[test])
                scores[preproc_id, fold_id, clf_id] = balanced_accuracy_score(y[test],y_pred)

    # Zapisanie  wyników 
    np.save('Results/results', scores)
