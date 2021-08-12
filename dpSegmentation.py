import numpy as np
from numba import jit, njit, prange
import time
from distance import euclidean
import pandas as pd
from utiles import dataCellSpecificLibrarySizeFactors



@njit(nopython=True)
def mean_numba(a):

    res = []
    for i in range(a.shape[1]):
        res.append(a[:, i].mean())

    return np.array(res)




def costOfSingleSegment(seg):
    (bins, features) = seg.shape
    center = np.mean(seg, axis=0)
    cost = 0.0
    for bin in range(bins):
        cost += euclidean(seg[bin, :].transpose(), center.transpose())

    return cost

@njit(nopython=True, fastmath=True)
def costOfSingleSeg(seg):
    (bins, features) = seg.shape
    center = mean_numba(seg)
    cost = 0.0
    for bin in range(bins):
        cost += euclidean(seg[bin, :].transpose(), center.transpose())

    return cost

@njit(nopython=True, parallel=True)
def calculateCostMatrix(data, K):
    bins = data.shape[0]
    costOfSegMetrix = np.zeros((bins, bins))+1e20
    for start in prange(bins):
        for end in prange(1, min(bins + 1, min(start+1000, start + 100 + 2*np.int(bins/K))), 1):
            if start < end:
                costOfSegMetrix[start][end - 1] = costOfSingleSeg(data[start:end, :])

    return costOfSegMetrix



class dpKmeans(object):
    def __init__(self, K):
        self.K = K



    def fit(self, data):
        (bins, features) = data.shape
        totalCost = np.zeros((bins, self.K+1)) + 1e20
        totalCost[:, 0] = 0.0
        finalSegLength = np.zeros((bins, self.K))

        costMatrix = calculateCostMatrix(data, self.K)



        for k in range(1, self.K+1, 1):
            time1 = time.time()
            for i in range(k, bins-(self.K - k) + 1):

                if k == 1:
                    totalCost[i-1][k] = costMatrix[0][i-1]
                    finalSegLength[i-1][k-1] = i

                if (k > 1 and k < self.K):

                    costPossibles = totalCost[k-2:i-1, k-1] + costMatrix[k-1:i, i-1]
                    totalCost[i-1][k] = np.min(costPossibles)
                    finalSegLength[i-1][k-1] = len(range(np.argmin(costPossibles) + k - 1, i, 1))



                if k == self.K:
                    if i < bins-1:
                        continue
                    if i == bins-1:

                        costPossibles = totalCost[k-2:i, k-1] + costMatrix[k-1:i+1, i]
                        totalCost[i][k] = np.min(costPossibles)
                        finalSegLength[i][k-1] = len(range(np.argmin(costPossibles)+k-1, i+1, 1))

            time2 = time.time()
            print("k = ", k, "time", time2-time1, "S")
        totalCost = totalCost[:, 1:]

        # determine the length of every segment
        lengthOfEverySeg = []
        lengthOfLastSeg = 0
        for k in range(self.K-1, -1, -1):
            if k == self.K - 1:
                lengthOfEverySeg.append(finalSegLength[bins - 1][k])
                lengthOfLastSeg = np.int(bins - 1 - finalSegLength[bins - 1][k])

            else:
                lengthOfEverySeg.append(finalSegLength[lengthOfLastSeg][k])
                lengthOfLastSeg = np.int(lengthOfLastSeg - finalSegLength[lengthOfLastSeg][k])

        lengthOfEverySeg = lengthOfEverySeg[::-1]
        return np.array(lengthOfEverySeg),totalCost

