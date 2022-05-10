import numpy as np

def makeDatasetReadable(datasetX, datasety, datasetname):
    X = np.genfromtxt(datasetX, delimiter=',')
    y = np.reshape(np.genfromtxt(datasety, delimiter=','), (X.shape[0],1))

    print(X,"\n", y)
    dataset = np.concatenate((X, y), axis=1)
    np.savetxt(datasetname, dataset)

makeDatasetReadable('X.csv', 'y.csv', 'datasetn1.csv')