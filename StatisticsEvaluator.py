import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
from ModifiedClusterCentroid import ModifiedClusterCentroids
from strlearn.metrics import balanced_accuracy_score

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
    'Linear SVC': LinearSVC()
}

# Metody undersampligu
preprocs = {
    'none': None,
    'RUS' : RandomUnderSampler(random_state=1234),
    'CC': ClusterCentroids(random_state=1234),
    'NM': NearMiss(version=1),
    'MCC': ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='DBSCAN', eps=2),
    'MCC-2': ModifiedClusterCentroids(CC_strategy='auto', cluster_algorithm='DBSCAN', eps=3),
    'MCC-3': ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='OPTICS', eps=2),
    'MCC-4': ModifiedClusterCentroids(CC_strategy='auto', cluster_algorithm='OPTICS', eps=2)

}

# Zbiór danych
#datasets = ['cpu_act','cpu_small','datatrieve', 'german','house_8L','kc1','kc2','kc3','schlvote','sick_numeric']
datasets = ['datatrieve']

if __name__ =='__main__':
    # Walidacja krzyzowa
    n_splits = 2
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1234)

    # Tablice z wynikami
    scores = np.zeros((len(datasets), len(preprocs), n_splits*n_repeats, len(clfs)))
    
    # Eksperyment
    for data_id, data_name in enumerate(datasets):
        dataset = np.genfromtxt("datasets/%s.csv" % (data_name) , delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for preproc_id, preproc_name in enumerate(preprocs):
                if preprocs[preproc_name] == None:
                    X_res, y_res = X[train], y[train]
                else:
                    X_res, y_res = preprocs[preproc_name].fit_resample(X[train],y[train])
                    print(preproc_name, ': ', X_res.shape)

                for clf_id, clf_name in enumerate(clfs):
                    clf = clfs[clf_name]
                    clf.fit(X_res, y_res)
                    y_pred = clf.predict(X[test])
                    scores[data_id, preproc_id, fold_id, clf_id] = balanced_accuracy_score(y[test],y_pred)

    #zapisanie  wyników 
    np.save('Results/statistic_results', scores)