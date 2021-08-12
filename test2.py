import numpy as np
import pandas as pd
from utiles import dataCellSpecificLibrarySizeFactors, dataBinSpecificBias, initialMatrice, LossFunction, chooseNegativeControlCells, diffOfTensor, diffOfBias
from negativeControl import negativeControlCellsIdentify
from optimize import optimize
import statsmodels.api as sm
from multiprocessing import Pool
import cv2
import time


def fGCOfSingleCell(temp):
    lowess = sm.nonparametric.lowess
    fgc = lowess(temp, list(range(len(temp))), frac=0.45)
    return fgc


def getFGC(Y, bins, cells, bias, G, H):
    biasMatrix = np.repeat(bias.reshape((bins, 1)), cells, axis=1)
    temp = Y / (biasMatrix * np.exp(np.dot(G, H)))
    pool = Pool(7)
    res = []
    for j in range(cells):
        res.append(pool.apply_async(fGCOfSingleCell, (temp[:, j],)))
    pool.close()
    pool.join()
    fGC = [x.get() for x in res]
    fGC = np.array(fGC)
    fGC[fGC <= 0] = 0.01
    return fGC[:, :, 1].transpose()


def getFGC1(Y, bins, cells, bias, G, H):
    biasMatrix = np.repeat(bias.reshape((bins, 1)), cells, axis=1)
    temp = Y / (biasMatrix * np.exp(np.dot(G, H)))
    fGC = []
    for j in range(cells):
        fGC.append(fGCOfSingleCell(temp[:, j]))
    fGC = np.array(fGC)
    fGC[fGC <= 0] = 0.01
    return fGC[:, :, 1].transpose()

if __name__ == "__main__":
    originFileName = "scope_experiment/T10_raw.csv"
    originData = pd.read_csv(originFileName, header=0, index_col=0)
    binsNames = originData._stat_axis.values.tolist()
    cellsNames = originData.columns.values.tolist()
    cellSpecificNormalizedOriginalDataFrame, sizeFactors = dataCellSpecificLibrarySizeFactors(originData)

    Y= cellSpecificNormalizedOriginalDataFrame.values
    (bins, cells) = Y.shape
    K = 50
    bias0 = np.ones((1, bins))
    biasOld = bias0
    G = np.zeros((bins, K))
    H = np.zeros((K, cells))
    t1 = time.time()
    fGC = getFGC(Y, bins, cells, biasOld, G, H)
    t2 = time.time()
    print(t2 - t1)
    # fGC1 = getFGC1(Y, bins, cells, biasOld, G, H)
    t3 = time.time()
    print(t3-t2)
    fGC2 = cv2.blur(Y, (1001,51))
    t4 = time.time()
    print(t4-t3)