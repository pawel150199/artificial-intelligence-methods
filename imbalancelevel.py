from tkinter import W
import matplotlib.pyplot as plt 
import numpy as np
import math

from sklearn import datasets

class Imbalance():
    def __init__(self, datasets):
        self.datasets = datasets
        self.classes_distrobution = []
        self.n_classes = []
        self.y = []
        self.dataset_name = []

    def calcutate(self):
        for data_id, dataset in enumerate(self.datasets):
            dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=',')
            self.dataset_name.append(dataset)
            self.y.append(dataset[:, -1].astype(int))
            self.n_classes.append(np.unique(self.y[data_id]))
            classes = np.unique(self.y[data_id])
            xd = []
            for i in classes:
                n = 0
                for j in self.y[data_id]:
                    if j == i:
                        n += 1
                xd.append(n)
            self.classes_distrobution.append(xd)

    def plot(self):
        x = len(self.datasets)
        w = int(math.ceil(x/3))
        if w == 1:
            fig, ax = plt.subplots(1,1, figsize=(8,8))
        else:
            fig, ax = plt.subplots(w,w, figsize=(8,8))
        xd = []
        for i in range(0,len(self.datasets)):
            for i in range (0,w):
                for j in range (0,w):
                    xd.append([i,j])
                


        for k in range(0, len(self.datasets)):
            i,j = xd[k]
            name = self.datasets[k]
            ax[i, j].bar(self.n_classes[k], self.classes_distrobution[k], width = 0.4)
            ax[i, j].set_title(f"{name}")
            ax[i, j].set_xlabel("Classes")
            ax[i, j].set_ylabel("Number of samples")
            ax[i, j].set_xlim(0,len(self.n_classes[k]))
            ax[i, j].set_ylim(0, len(self.y[k]))
        
        plt.tight_layout()
        plt.show()
        
if __name__=="__main__":
    x = Imbalance(['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit', 'ecoli4', 'german', 'glass2'])
    x.calcutate()
    x.plot()




        
                        









        
