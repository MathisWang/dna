import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import random
from sklearn.metrics import mean_squared_error as mse
from utiles import dataCellSpecificLibrarySizeFactors
import time
from multiprocessing import Pool

#1

# def dataBinSpecificBias(negativeCellsDataFrame, G, H):
#     GH = np.dot(G, H)
#     beltaEstimatedMatrix = negativeCellsDataFrame.values/np.exp(GH)
#     belta = np.median(beltaEstimatedMatrix, axis=1)
#     return belta
#
# a = np.array([[10,3,7],[1,2,8]])
# b = [1, 2]q
# b = np.array(b)
# b = b.reshape((2,1))
#
# c = np.repeat(b, 3, axis=1)
# d = c*a
# print(c)
# print(d)

originFileName = "scope_experiment/A_raw.csv"
originData = pd.read_csv(originFileName, header=0, index_col=0)
binsNames = originData._stat_axis.values.tolist()
cellsNames = originData.columns.values.tolist()
cellSpecificNormalizedOriginalDataFrame, sizeFactors = dataCellSpecificLibrarySizeFactors(originData)

Y = cellSpecificNormalizedOriginalDataFrame.values


lowess=sm.nonparametric.lowess
ySm = lowess(Y[:,3], list(range(Y.shape[0])), frac=0.02)
print(ySm.shape)
print(ySm)
plt.plot(Y[:,3])
plt.plot(ySm[:,1])
plt.show()
#

# def getFGC(Y, bins, cells, bias, G, H):
#     fGC = []
#     biasMatrix = np.repeat(bias.reshape((bins, 1)), cells, axis=1)
#     temp = Y/(biasMatrix*np.exp(np.dot(G, H)))
#     lowess = sm.nonparametric.lowess
#     for j in range(cells):
#         fGCTemp = lowess(temp[:,j], list(range(bins)), frac=0.01)
#         fGC.append(fGCTemp[:, 1])
#         print(j)
#     return np.array(fGC)

# def fGCOfSingleCell(temp):
#     lowess = sm.nonparametric.lowess
#     fgc = lowess(temp, list(range(len(temp))), frac=0.01)
#     return fgc
#
# def getFGC(Y, bins, cells, bias, G, H):
#     biasMatrix = np.repeat(bias.reshape((bins, 1)), cells, axis=1)
#     temp = Y / (biasMatrix * np.exp(np.dot(G, H)))
#     pool = Pool(8)
#     res = []
#     for j in range(cells):
#         res.append(pool.apply_async(fGCOfSingleCell, (temp[:, j],)))
#     pool.close()
#     pool.join()
#     fGC = [x.get() for x in res]
#     fGC = np.array(fGC)
#     return fGC[:, :, 1].transpose()
#
#
#
#
# K = 50
# (bins, cells) = Y.shape
# bias0 = np.ones((1, bins))
# biasOld = bias0
# G = np.zeros((bins, K))
# H = np.zeros((K, cells))
# biasMatrix = np.repeat(bias0.reshape((bins, 1)), cells, axis=1)
# temp = Y/(biasMatrix*np.exp(np.dot(G, H)))
#
#
#
#
#
#
#
#
#
#
# if __name__ == "__main__":
#     fGC = getFGC(Y, bins, cells, bias0, G, H)
#     print(fGC)
