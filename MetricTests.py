import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from ModifiedClusterCentroid import ModifiedClusterCentroids
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from MetricEvaluate import mrts, preprocs, datasets
""" Klasyfikatory uzyte w eksperymencie
    GNB: GaussianNB
    SVC: SVC
    kNN: KNeighborsClassifier
    Linear SVC: LinearSVC
"""

#global variable used for statistics tests
alpha=.05
m_fmt="%.3f"
std_fmt=None
nc="---"
db_fmt="%s"
tablefmt="plain"

if __name__=="__main__":
    #import wyników
    scores = np.load("metric_results_GNB.npy")
    mean_scores = np.mean(scores, axis=2)
    stds = np.std(scores, axis=2)
    clfs = list(preprocs.keys())
    n_clfs = len(preprocs)
    # Perform tests
    tables = {}
    for m_idx, m_name in enumerate(preprocs):
        # Prepare storage for table
        t = []
        for db_idx, db_name in enumerate(datasets):
            # Row with mean scores
            t.append([db_fmt % db_name] + [m_fmt % v for v in mean_scores[db_idx, m_idx, :]])
            # Row with std
            if std_fmt:
                t.append([''] + [std_fmt %
                                 v for v in
                                 stds[db_idx, :, m_idx]])
            # Calculate T and p
            T, p = np.array(
                [[ttest_ind(scores[db_idx, m_idx, i, :],
                       scores[db_idx, m_idx, j, :])
                  for i in range(len(clfs))]
                 for j in range(len(clfs))]
            ).swapaxes(0, 2)
            _ = np.where((p < alpha) * (T > 0))
            conclusions = [list(1 + _[1][_[0] == i])
                           for i in range(n_preprocs)]
            # Row with conclusions
            # t.append([''] + [", ".join(["%i" % i for i in c])
            #                  if len(c) > 0 else nc
            #                  for c in conclusions])
            t.append([''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < len(preprocs)-1 else ("all" if len(c) == len(preprocs)-1 else nc)
                             for c in conclusions])
        # Store formatted table
        tables.update({m_name: tabulate(
            t, headers=['DATASET'] + preprocs, tablefmt=tablefmt)})