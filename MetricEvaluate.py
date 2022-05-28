import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from ModifiedClusterCentroid import ModifiedClusterCentroids
from strlearn.metrics import precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score

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
    'RUS' : RandomUnderSampler(),
    'CC': ClusterCentroids(random_state=1234),
    #'MCC': ModifiedClusterCentroids(CC_strategy='const')
}
# Metryki
mrts = {
    'specificity': specificity,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

# Zbiór danych
datasets = ['yeast6', 'balance', 'australian']

if __name__ =='__main__':
    # Walidacja krzyzowa
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1234)

    # Tablice z wynikami
    scores = np.zeros((len(datasets), len(preprocs), n_splits*n_repeats, len(mrts), len(clfs)))
    
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

                for clf_id, clf_name in enumerate(clfs):
                    clf_GNB = clfs[clf_name]
                    clf_GNB.fit(X_res, y_res)
                    y_pred = clf_GNB.predict(X[test])
                    for m_id, m_name in enumerate(mrts):
                        mtr = mrts[m_name]
                        scores[data_id, preproc_id, fold_id, m_id, clf_id] = mtr(y[test],y_pred)

    #zapisanie  wyników 
    np.save('Results/metric_results', scores)
