import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from ModifiedClusterCentroid import ModifiedClusterCentroids
from StatisticsEvaluator import preprocs, clfs

""" 
Przeprowadzono testy statystyczne

Klasyfikatory uzyte w eksperymencie:
    *GNB: GaussianNB
    *SVC: SVC
    *kNN: KNeighborsClassifier
    *Linear SVC: LinearSVC
"""

for clf_id, clf_name in enumerate(clfs):
    #import wyników
    scores = np.load("Results/results.npy")
    scores = scores[:, :, clf_id]

    # Przedział ufności i tablice na wyniki
    alfa = .05
    t_statistic = np.zeros((len(preprocs), len(preprocs)))
    p_value = np.zeros((len(preprocs), len(preprocs)))

    for i in range(len(preprocs)):
        for j in range(len(preprocs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])

    # Wartości P  i T
    headers = list(preprocs.keys())
    names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("\n\n\tStatistic tests for GNB classificator")
    print("\n\nGNB t-statistic:\n\n", t_statistic_table, "\n\n GNB p-value:\n\n", p_value_table)

    # Tablica przewag
    advantage = np.zeros((len(preprocs), len(preprocs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\n\nAdvantage: \n\n", advantage_table)

    # Róznice statystyczne znaczące
    significance = np.zeros((len(preprocs), len(preprocs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
    print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

    # Wyniki koncowe analizy statystycznej
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
    print("\n\nStatistically significantly better:\n\n", stat_better_table)
