import numpy as np
from tabulate import tabulate

class IR:
    "Klasa słuzy do wygenerowania tabeli ze statystykami zbiorów danych"
    def __init__(self, datasets):
        self.datasets = datasets
    
    def calculate(self):
        self.scores = []
        self.y = []
        self.dataset_name = []
        for data_id, dataset in enumerate(self.datasets):
            dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=',')
            self.y = dataset[:, -1].astype(int)
            l, c = np.unique(self.y, return_counts=True)
            minor_probas = np.amin(c)
            idx = np.where(minor_probas!=c)
            Nmax = sum(c[idx])
            Nmin = minor_probas
            IR = round((Nmax/Nmin), 2)
            self.scores.append([Nmin, Nmax, IR])
        return self

    def tab(self):
        t = []
        for data_id, data_name in enumerate(self.datasets):
            t.append(['%s' % data_name] + ['%.3f' % v for v in self.scores[data_id]])
        headers = ['datasets', 'N_min', 'N_maj', 'IR']
        print(tabulate(t, headers))
        #with open('LatexTable/Dataset_descripttion.txt', 'w') as f:
            #f.write(tabulate(t, headers, tablefmt='latex'))

if __name__=='__main__':
    xd = IR(['glass1', 'wisconsin', 'pima', 'iris0', 'glass0', 'yeast1', 'haberman', 'vehicle2', 'vehicle3', 'ecoli1', 'segment0', 'glass6'])
    xd.calculate()
    xd.tab()
