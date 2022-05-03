import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata, ranksums
from tabulate import tabulate
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

#Wartości średnie z wyników. Srednia liczona po foldach
mean_scores_GNB = np.mean(scores_GNB, axis=2).T
mean_scores_SVC = np.mean(scores_SVC, axis=2).T
mean_scores_kNN = np.mean(scores_kNN, axis=2).T
mean_scores_LSVC = np.mean(scores_LSVC, axis=2).T

#________________________TESTY RANKINGOWE_____________________

alfa = .05
w_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

#GNB
ranks_GNB = []
for ms in mean_scores_GNB:
    ranks_GNB.append(rankdata(ms).tolist())
ranks_GNB = np.array(ranks_GNB)

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_GNB.T[i], ranks_GNB.T[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\n\tRanking tests for GNB classificator")
print("w-statistic:\n\n", w_statistic_table, "\n\np-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Rónice statystyczne znaczące
significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)


#SVC
ranks_SVC = []
for ms in mean_scores_SVC:
    ranks_SVC.append(rankdata(ms).tolist())
ranks_SVC = np.array(ranks_SVC)

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_SVC.T[i], ranks_SVC.T[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\n\tRanking tests for SVC classificator")
print("w-statistic:\n\n", w_statistic_table, "\n\np-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Rónice statystyczne znaczące
significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)

#kNN
ranks_kNN = []
for ms in mean_scores_kNN:
    ranks_kNN.append(rankdata(ms).tolist())
ranks_kNN = np.array(ranks_kNN)

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_kNN.T[i], ranks_kNN.T[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\n\tRanking tests for kNN classificator")
print("w-statistic:\n\n", w_statistic_table, "\n\np-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Rónice statystyczne znaczące
significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)

#LSVC
ranks_LSVC = []
for ms in mean_scores_LSVC:
    ranks_LSVC.append(rankdata(ms).tolist())
ranks_LSVC = np.array(ranks_LSVC)

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_LSVC.T[i], ranks_LSVC.T[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\n\tRanking tests for LSVC classificator")
print("w-statistic:\n\n", w_statistic_table, "\n\np-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Rónice statystyczne znaczące
significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)
