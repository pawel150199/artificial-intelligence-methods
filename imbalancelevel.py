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
        self.xticks = []
        for data_id, dataset in enumerate(self.datasets):
            dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=',')
            self.dataset_name.append(dataset)
            self.y.append(dataset[:, -1].astype(int))
            self.n_classes.append(np.unique(self.y[data_id]))
            classes = np.unique(self.y[data_id])
            self.xticks.append(classes)
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

        if x <= 2:
            fig, ax = plt.subplots(1,2, figsize=(10,5))
            for i in range(0,x):
                name = self.datasets[i]
                ax[i-1].bar(self.n_classes[i], self.classes_distrobution[i], width = 0.4)
                ax[i-1].set_title(f"{name}")
                ax[i-1].set_xlabel("Classes")
                ax[i-1].set_ylabel("Number of samples")
                ax[i-1].set_xlim(-1,len(self.n_classes[i]))
                ax[i-1].set_ylim(0, len(self.y[i]))
                ax[i-1].set_xticks(self.xticks[i])
                
                
        
            
        else:
            fig, ax = plt.subplots(w,w+1, figsize=(10,5))
            xd = []
            for i in range (0,w):
                for j in range (0,w+1):
                    print(w)
                    print(i,j)
                    xd.append([i,j])
                    
            for k in range(0, len(self.datasets)):
                i,j = xd[k]
                name = self.datasets[k]
                ax[i, j].bar(self.n_classes[k], self.classes_distrobution[k], width = 0.4)
                ax[i, j].set_title(f"{name}")
                ax[i, j].set_xlabel("Classes")
                ax[i, j].set_ylabel("Number of samples")
                ax[i, j].set_xlim(-1,len(self.n_classes[k]))
                ax[i, j].set_ylim(0, len(self.y[k]))
                ax[i, j].set_xticks(self.xticks[k])
        
        plt.tight_layout()
        plt.show()
