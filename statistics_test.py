import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata
from tabulate import tabulate
from scipy.stats import ttest_ind
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

""" Klasyfikatory uzyte w eksperymencie
    GNB: GaussianNB
    SVC: SVC
    kNN: KNeighborsClassifier
    Linear SVC: LinearSVC"""

preprocs = {
    'none': None,
    'RUS' : RandomUnderSampler(),
    'CC': ClusterCentroids(random_state=1234) 
}

#import wyników
scores_GNB = np.load("results_GNB.npy")
scores_SVC = np.load("results_SVC.npy")
scores_kNN = np.load("results_kNN.npy")
scores_LSVC = np.load("results_LSVC.npy")

#________________________TESTY STATYSTYCZNE_____________________
"""Przeprowadzono testy statystyczne w celu określenia, która metoda preprocesingu jest najlepszana określonym klasyfikatorze"""
alfa = .05
t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

#GNB
mean_scores_GNB = np.mean(scores_GNB, axis=1)
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(mean_scores_GNB[i], mean_scores_GNB[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\n\tStatistic tests for GNB classificator")
print("\n\nGNB t-statistic:\n\n", t_statistic_table, "\n\n GNB p-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Róznice statystyczne znaczące
significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)

#SVC
mean_scores_SVC = np.mean(scores_SVC, axis=1)
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(mean_scores_SVC[i], mean_scores_SVC[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\n\tStatistic tests for SVC classificator\n\n")
print("\n\nSVC t-statistic:\n\n", t_statistic_table, "\n\n SVC p-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Róznice statystyczne znaczące
significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)

#kNN
mean_scores_kNN = np.mean(scores_kNN, axis=1)
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(mean_scores_kNN[i], mean_scores_kNN[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\n\tStatistic tests for kNN classificator")
print("\n\nkNN t-statistic:\n\n", t_statistic_table, "\n\n kNN p-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Róznice statystyczne znaczące
significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)

#LSVC
mean_scores_LSVC = np.mean(scores_LSVC, axis=1)
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(mean_scores_LSVC[i], mean_scores_LSVC[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\n\tStatistics tests for  Linear SVC classificator")
print("\n\nLinear SVC t-statistic:\n\n", t_statistic_table, "\n\n Linear SVC p-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Róznice statystyczne znaczące
significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)
