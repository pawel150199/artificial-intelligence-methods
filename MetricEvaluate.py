import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from ModifiedClusterCentroid import ModifiedClusterCentroids
from strlearn.metrics import precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score

""" Klasyfikatory uzyte w eksperymencie
    GNB: GaussianNB
    SVC: SVC
    kNN: KNeighborsClassifier
    Linear SVC: LinearSVC
"""

#metody undersampligu
preprocs = {
    'none': None,
    'RUS' : RandomUnderSampler(),
    'CC': ClusterCentroids(random_state=1234),
    #'MCC': ModifiedClusterCentroids(CC_strategy='const')
}
#metryki
mrts = {
    'specificity': specificity,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

#walidacja krzyzowa
n_splits = 5
n_repeats = 2
datasets = ['yeast6', 'balance', 'australian']
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1234)

#tablice z wynikami
scores_GNB = np.zeros((len(datasets), len(preprocs), n_splits*n_repeats, len(mrts)))
scores_SVC = np.zeros((len(datasets), len(preprocs), n_splits*n_repeats, len(mrts)))
scores_kNN = np.zeros((len(datasets), len(preprocs), n_splits*n_repeats, len(mrts)))
scores_LSVC = np.zeros((len(datasets), len(preprocs), n_splits*n_repeats, len(mrts)))

#Eksperyment
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

            #GNB
            clf_GNB = GaussianNB()
            clf_GNB.fit(X_res, y_res)
            y_pred = clf_GNB.predict(X[test])
            for m_id, m_name in enumerate(mrts):
                mtr = mrts[m_name]
                scores_GNB[data_id, preproc_id, fold_id, m_id] = mtr(y[test],y_pred)

            #SVC
            clf_SVC = SVC()
            clf_SVC.fit(X_res, y_res)
            y_pred = clf_SVC.predict(X[test])
            for m_id, m_name in enumerate(mrts):
                mtr = mrts[m_name]
                scores_GNB[data_id, preproc_id, fold_id, m_id] = mtr(y[test],y_pred)

            #kNN
            clf_kNN = KNeighborsClassifier()
            clf_kNN.fit(X_res, y_res)
            y_pred = clf_kNN.predict(X[test])
            for m_id, m_name in enumerate(mrts):
                mtr = mrts[m_name]
                scores_GNB[data_id, preproc_id, fold_id, m_id] = mtr(y[test],y_pred)

                
            #LSVC
            clf_LSVC = LinearSVC(random_state=1234, tol=1e-5)
            clf_LSVC.fit(X_res, y_res)
            y_pred = clf_LSVC.predict(X[test])
            for m_id, m_name in enumerate(mrts):
                mtr = mrts[m_name]
                scores_GNB[data_id, preproc_id, fold_id, m_id] = mtr(y[test],y_pred)

#zapisanie  wyników 
np.save('metric_results_GNB', scores_GNB)
np.save('metric_results_SVC', scores_SVC)
np.save('metric_results_kNN', scores_kNN)
np.save('metric_results_LSVC', scores_LSVC)
