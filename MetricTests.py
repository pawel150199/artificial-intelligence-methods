from matplotlib.pyplot import axis
import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind
from MetricEvaluate import mrts, preprocs, datasets, clfs
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
metrics = list(mrts.keys())
preprocs = list(preprocs.keys())
n_preprocs = len(preprocs)

if __name__=="__main__":
    #import wyników

    for clf_id, clf_name in enumerate(clfs):
        scores = np.load("metric_results.npy")
        scores = scores[:,:,:,:,clf_id]
        mean_scores = np.mean(scores, axis=2)
        stds = np.std(scores, axis=2)

        # Perform tests
        tables = {}
        t = []
        for m_idx, m_name in enumerate(metrics):
            # Prepare storage for table
            for db_idx, db_name in enumerate(datasets):
                # Row with mean scores
                t.append(['%s' % m_name]+ [db_fmt % db_name] + [m_fmt % v for v in mean_scores[db_idx, :, m_idx]])
                # Row with std
                if std_fmt:
                    t.append(['']+[''] + [std_fmt % v for v in stds[db_idx, :, m_idx]])
                # Calculate T and p
                T, p = np.array(
                    [[ttest_ind(scores[db_idx, i, :, m_idx],
                        scores[db_idx, j, :, m_idx])
                    for i in range(len(preprocs))]
                    for j in range(len(preprocs))]
                ).swapaxes(0, 2)
                _ = np.where((p < alpha) * (T > 0))
                conclusions = [list(1 + _[1][_[0] == i])
                            for i in range(n_preprocs)]
        
                t.append(['']+[''] + [", ".join(["%i" % i for i in c])
                                if len(c) > 0 and len(c) < len(preprocs)-1 else ("all" if len(c) == len(preprocs)-1 else nc)
                                for c in conclusions])

            # Store formatted table
        print('\n\n\n', clf_name, '\n')  
        headers = ['metrics', 'datasets']
        for i in preprocs:
            headers.append(i)
        print(headers)
        print(tabulate(t))
        