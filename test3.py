import numpy as np
import pandas as pd
import re
from dpSegmentation import dpKmeans
dataFile = "MFResult/T16P_leftMatrix_Fano.csv"
dataFrame = pd.read_csv(dataFile,header=0, index_col=0)
binsNames = dataFrame._stat_axis.values.tolist()


# seg1 = []
#
#
# seg1 = np.array(seg1)
data = dataFrame.values
#
# seg = dpKmeans(K=11)
# a, cost = seg.fit(data[min(seg1):max(seg1) + 1, :])
# binsNom = np.zeros((len(a), 2))
# print(a)

# print(binsNom)

binsSegmentation = np.zeros((22*11, 2))
for n in range(1, 23, 1):
    l = []
    for i, b in enumerate(binsNames):
        if re.match("chr" + str(n) + ":.*", b):
            l.append(i)
    seg = dpKmeans(K=11)
    a, cost = seg.fit(data[min(l):max(l) + 1, :])
    binsNom = np.zeros((len(a), 2))
    start = 0
    for i in range(len(a)):
        binsNom[i, 0] = start
        binsNom[i, 1] = start + a[i] - 1
        start = start + a[i]

    binsSegmentation[(n-1)*11: n*11, :] = binsNom

print(binsSegmentation)

rowNames = np.array([("chr"+str(n),)*11 for n in range(1, 23)]).reshape((22*11,1))

res = pd.DataFrame(binsSegmentation, columns=["StartBin", "EndBin"], index=rowNames)

res.to_csv("segment/T16P_segments_ourModel.csv")