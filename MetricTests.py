from matplotlib.pyplot import axis
import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind
from MetricEvaluate import mrts, preprocs, datasets, clfs

""" Wygenerowano tablice z wynikami pochodzącymi z MetricEvaluate

Klasyfikatory uzyte w eksperymencie:
    *GNB: GaussianNB
    *SVC: SVC
    *kNN: KNeighborsClassifier
    *Linear SVC: LinearSVC


"""

# Zmienne globalne uzyte w testach statystycznych
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
    # Generowanie tabel
    for clf_id, clf_name in enumerate(clfs):
        # Pobranie wyników
        scores = np.load("Results/metric_results.npy")
        scores = scores[:,:,:,:,clf_id]
        mean_scores = np.mean(scores, axis=2)
        stds = np.std(scores, axis=2)
        t = []

        for m_idx, m_name in enumerate(metrics):
            for db_idx, db_name in enumerate(datasets):
                # Wiersz z wartoscia srednia
                t.append(['%s' % m_name]+ [db_fmt % db_name] + [m_fmt % v for v in mean_scores[db_idx, :, m_idx]])
                # Jesli podamy std_fmt w zmiennych globalnych zostanie do tabeli dodany wiersz z odchyleniem standardowym
                if std_fmt:
                    t.append(['']+[''] + [std_fmt % v for v in stds[db_idx, :, m_idx]])
                # Obliczenie wartosci T i P
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

        # Prezentacja wyników
        print('\n\n\n', clf_name, '\n')  
        headers = ['metrics', 'datasets']
        for i in preprocs:
            headers.append(i)
        print(headers)
        print(tabulate(t))

        # Zapisanie wyników w formacie .tex
        with open('LatexTable/Statistic_%s.txt' % (clf_name), 'w') as f:
            f.write(tabulate(t, tablefmt='latex'))
        