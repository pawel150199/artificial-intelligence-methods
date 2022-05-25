import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from zad63 import RandomSubspaceEnsemble
from zad62 import BaggingClassifier2
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import warnings
from tabulate import tabulate
from scipy.stats import rankdata, ranksums

results=[]
datasets = ['banana', 'balance', 'appendicitis', 'iris']

m_fmt="%.3f"
std_fmt=None
nc="---" 
db_fmt="%s" 
tablefmt="plain"
alpha = 0.5


clfs = {
   'Bagging HV w_on': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=44), hard_voting=True, weight_mode=True, random_state=44),
    'Bagging HV w_off': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=44), hard_voting=True, weight_mode=False, random_state=44),
    'Bagging NHV w_on': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=44), hard_voting=False, weight_mode=True, random_state=44),
    'Bagging NHV w_off': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=44), hard_voting=False, weight_mode=False, random_state=44),
    #'RSM HV': RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=44), hard_voting=True, random_state=44, n_subspace_features=2),
    #'RSM NHV': RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=44), hard_voting=False, random_state=44, n_subspace_features=2)
}   
# Perform tests
test = np.zeros((len(clfs), len(clfs)))
tables = {}
t = []
scores= np.load("results2.npy")
for db_idx, db_name in enumerate(datasets):
    
    data_scores = scores[:,db_idx,:]
    t.append(["%s" % db_name] + ["%.3f" %
    v for v in np.mean(data_scores, axis=1)])
    t.append(["std"] + ["%.3f" %
    v for v in np.std(data_scores, axis=1)])
    
    t.append([''] + [", ".join(["%i" % i for i in c])
    if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)]) 

print(tabulate(t))

