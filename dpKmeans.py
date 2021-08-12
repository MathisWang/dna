import numpy as np
from distance import euclidean
import pandas as pd
from utiles import dataCellSpecificLibrarySizeFactors

import time
import numba


# @numba.njit()
def costOfSingleSegment(seg):
    (bins, features) = seg.shape
    center = np.mean(seg, axis=0)
    cost = 0.0
    for bin in range(bins):
        cost += euclidean(seg[bin, :].transpose(), center.transpose())

    return cost




class dpKmeans(object):
    def __init__(self, K):
        self.K = K



    def fit(self, data):
        (bins, features) = data.shape
        totalCost = np.zeros((bins, self.K+1)) + 1e20
        totalCost[:, 0] = 0.0
        finalSegLength = np.zeros((bins, self.K))

        time0 = time.time()
        costOfSegMetrix = np.zeros((bins, bins))
        for start in range(bins):
            for end in range(1, bins+1, 1):
                if start < end:
                    costOfSegMetrix[start][end - 1] = costOfSingleSegment(data[start:end, :])
        time_ = time.time()
        print(time_ - time0)
        print(costOfSegMetrix)



        for k in range(1, self.K+1, 1):
            time1 = time.time()
            for i in range(k, bins-(self.K - k) + 1):

                if k == 1:
                    totalCost[i-1][k] = costOfSingleSegment(data[0:i, :])
                    finalSegLength[i-1][k-1] = i

                if (k > 1 and k < self.K):
                    costPossibles = []

                    for t in range(k-1, i):
                        costPossible = totalCost[t-1][k-1] + costOfSingleSegment(data[t:i, :])
                        costPossibles.append(costPossible)

                    totalCost[i-1][k] = np.min(costPossibles)
                    finalSegLength[i-1][k-1] = len(range(np.argmin(costPossibles) + k - 1, i, 1))



                if k == self.K:
                    if i < bins-1:
                        continue
                    if i == bins-1:
                        costPossibles = []

                        for t in range(k - 1, i+1):
                            costPossible = totalCost[t-1][k - 1] + costOfSingleSegment(data[t:i+1, :])
                            costPossibles.append(costPossible)

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






#
# a = np.zeros((3,4))
# print(a[1:2,:])

originFileName = "scope_experiment/T10_raw.csv"
originData = pd.read_csv(originFileName, header=0, index_col=0)
cellSpecificNormalizedDataFrame, sizeFactors = dataCellSpecificLibrarySizeFactors(originData)
print(cellSpecificNormalizedDataFrame)

data = cellSpecificNormalizedDataFrame.values[0:100, :]
t1 = time.time()
seg = dpKmeans(K=8)
res, cost = seg.fit(data)
print(res)
t2 = time.time()
print(t2 - t1)
print(cost)