from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from sklearn.base import clone
from scipy.stats import rankdata
from tabulate import tabulate

clfs = {
    'GNB': GaussianNB(),
    'SVC': SVC(),
    'kNN': KNeighborsClassifier(),
    'Linear SVC': LinearSVC()
}
datasets = ['australian', 'balance', 'breastcan']

preprocs = {
    'none': None,
    'RUS' : RandomUnderSampler(),
    'CC': ClusterCentroids(random_state=1234) 
}

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1234)

scores = np.zeros((len(clfs), n_datasets, n_splits*n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=',')
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
                scores[clf_id, data_id, fold_id, preproc_id] = accuracy_score(y[test], y_pred)


mean = np.   
mean_scores = np.mean(scores, axis=2).T
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n", mean_ranks)



