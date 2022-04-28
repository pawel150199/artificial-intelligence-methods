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
from scipy.stats import ttest_rel

""" Klasyfikatory uzyte w eksperymencie
    GNB: GaussianNB
    SVC: SVC
    kNN: KNeighborsClassifier
    Linear SVC: LinearSVC"""

#zbiory danych
datasets = ['australian']

#metody undersampligu
preprocs = {
    'none': None,
    'RUS' : RandomUnderSampler(),
    'CC': ClusterCentroids(random_state=1234) 
}

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2

#walidacja krzyzowa
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1234)

#tablice z wynikami
scores_GNB = np.zeros((len(preprocs), n_datasets, n_splits*n_repeats))
scores_SVC = np.zeros((len(preprocs), n_datasets, n_splits*n_repeats))
scores_kNN = np.zeros((len(preprocs), n_datasets, n_splits*n_repeats))
scores_LSVC = np.zeros((len(preprocs), n_datasets, n_splits*n_repeats))

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

            #GNB
            clf_GNB = GaussianNB()
            clf_GNB.fit(X_res, y_res)
            y_pred = clf_GNB.predict(X[test])
            scores_GNB[preproc_id, data_id, fold_id] = accuracy_score(y[test],y_pred)
            #SVC
            clf_SVC = SVC()
            clf_SVC.fit(X_res, y_res)
            y_pred = clf_SVC.predict(X[test])
            scores_SVC[preproc_id, data_id, fold_id] = accuracy_score(y[test],y_pred)
            #kNN
            clf_kNN = KNeighborsClassifier()
            clf_kNN.fit(X_res, y_res)
            y_pred = clf_kNN.predict(X[test])
            scores_kNN[preproc_id, data_id, fold_id] = accuracy_score(y[test],y_pred)
            #LSVC
            clf_LSVC = LinearSVC(random_state=1234, tol=1e-5)
            clf_LSVC.fit(X_res, y_res)
            y_pred = clf_LSVC.predict(X[test])
            scores_LSVC[preproc_id, data_id, fold_id] = accuracy_score(y[test],y_pred)

#zapisanie  wyników 
np.save('results_GNB', scores_GNB)
np.save('results_SVC', scores_SVC)
np.save('results_kNN', scores_kNN)
np.save('results_LSVC', scores_LSVC)

#________________________TESTY STATYSTYCZNE_____________________
alfa = .05
t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

#GNB
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores_GNB[i], scores_GNB[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("GNB t-statistic:\n\n", t_statistic_table, "\n\n GNB p-value:\n\n", p_value_table)

#SVC
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores_SVC[i], scores_SVC[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("SVC t-statistic:\n\n", t_statistic_table, "\n\n SVC p-value:\n\n", p_value_table)

#kNN
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores_kNN[i], scores_kNN[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("kNN t-statistic:\n\n", t_statistic_table, "\n\n kNN p-value:\n\n", p_value_table)

#LSVC
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores_LSVC[i], scores_LSVC[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("Linear SVC t-statistic:\n\n", t_statistic_table, "\n\n Linear SVC p-value:\n\n", p_value_table)