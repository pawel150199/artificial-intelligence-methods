import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from ModifiedClusterCentroid import ModifiedClusterCentroids
from imbalancelevel import Imbalance
from sklearn.base import clone

clfs = {
    'GNB': GaussianNB(),
    'SVC': SVC(),
    'kNN': KNeighborsClassifier(),
    'Linear SVC': LinearSVC()
}

#metody undersampligu
preprocs = {
    'none': None,
    'RUS' : RandomUnderSampler(),
    'CC': ClusterCentroids(random_state=1234),
    #'MCC': ModifiedClusterCentroids(CC_strategy='const')
}

#walidacja krzyzowa
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1234)

datasets = ['yeast6', 'yeast5', 'yeast3', 'wine']
#dataset = np.genfromtxt("datasets/%s.csv" % (datasets) , delimiter=',')
#ap = Imbalance(datasets)
#ap.calcutate()
#ap.plot()
#tablice z wynikami
scores = np.zeros((len(datasets), len(preprocs), len(clfs), n_splits*n_repeats))

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
                clf = clone(clfs[clf_name])
                clf.fit(X_res, y_res)
                y_pred = clf.predict(X[test])
                scores[data_id, preproc_id, clf_id, fold_id] = balanced_accuracy_score(y[test],y_pred)

            

#zapisanie  wyników 
np.save('results', scores)

