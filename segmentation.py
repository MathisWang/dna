import pandas as pd
import numpy as np
from utiles import dataCellSpecificLibrarySizeFactors, dataBinSpecificBias, initialMatrice, LossFunction, chooseNegativeControlCells, diffOfTensor
from negativeControl import negativeControlCellsIdentify
from optimize import optimize
from multiprocessing import Pool
import statsmodels.api as sm


originFileName = "scope_experiment/T10_raw.csv"
originData = pd.read_csv(originFileName, header=0, index_col=0)
binsNames = originData._stat_axis.values.tolist()
cellsNames = originData.columns.values.tolist()
cellSpecificNormalizedOriginalDataFrame, sizeFactors = dataCellSpecificLibrarySizeFactors(originData)


normalizedFileName = "result/T10_nor_demo.csv"
normalizedData = pd.read_csv(normalizedFileName, header=0, index_col=0)
cellSpecificNormalizedNorDataFrame, sizeFactors1 = dataCellSpecificLibrarySizeFactors(normalizedData)

biasFileName = "result/T10_ref_demo2.csv"
biasData = pd.read_csv(biasFileName, header=0, index_col=0)
Y = cellSpecificNormalizedOriginalDataFrame.values
mu = cellSpecificNormalizedNorDataFrame.values
bias = biasData.values
# fGCFileName = "result/T10_fGC_demo2.csv"
# fGCData = pd.read_csv(fGCFileName, header=0, index_col=0)


def intializetionOfPloidy(Y, mu, minPloidyNum, maxPloidyNum):
    mu = mu + 0.01
    res = []
    (bins, cells) = Y.shape
    P = np.linspace(minPloidyNum, maxPloidyNum, num=(np.int((maxPloidyNum-minPloidyNum)/0.05)))
    for j in range(cells):
        diff = list(map(lambda x: sum(((Y[:, j]*x/mu[:, j])-(np.round(Y[:, j]*x/mu[:, j])))**2), P))
        p = P[np.argmin(diff)]
        res.append(p)
    return np.array(res)


def Mstep(Y, Z, pi, G, H, Bb, Bc, bias, negativeControlCells, indiceOfNegativeControleCells, minCopyNum, K, steps, iterationNum, eps, alpha, batchSize):
    (bins, cells, groups) = Z.shape
    for j in range(cells):
        for t in range(groups):
            pi[j][t] = sum(Z[:, j, t])/bins

    for step in range(steps):
        biasNew = dataBinSpecificBias(negativeControlCells, indiceOfNegativeControleCells, G, H)


        biasNewMatrix = np.repeat(biasNew.reshape((bins, 1)), cells, axis=1)

        a = np.log(Y+0.01/biasNewMatrix)
        G, H = initialMatrice(a, nFeatures=K)

        lossFuncs = []
        lossFuncPre = float("-inf")

        for i in range(iterationNum):
            mu = biasNewMatrix*np.exp(np.dot(G, H))
            mu[mu <= 0] = 0
            B = np.dot(Bb, Bc)
            B[B <= 0] = 1e-10
            lossFunc = LossFunction(mu, B, Y)
            if abs(lossFunc - lossFuncPre) <= eps:
                lossFuncs.append(lossFunc)
                break
            else:
                lossFuncPre = lossFunc
                lossFuncs.append(lossFunc)
                binsChosen = np.random.randint(low=0, high=bins, size=batchSize)
                negativeControlCellsChosen = chooseNegativeControlCells(indiceOfNegativeControleCells, batchSize)
                for i, j in zip(binsChosen, negativeControlCellsChosen):
                    dG, dH, dBb, dBc = optimize(mu, B, Y, G, H, Bb, Bc, i, j)
                    G[i, :] = G[i, :] + alpha*dG
                    H[:, j] = H[:, j] + alpha*dH
                    Bb[i, :] = Bb[i, :] + alpha*dBb
                    Bc[:, j] = Bc[:, j] + alpha*dBc
    # sumMatrix = np.zeros((bins, cells))
    # biasNewMatrix = np.repeat(bias.reshape((bins, 1)), cells, axis=1)
    # for t in range(groups):
    #     sumMatrix += biasNewMatrix*((t+minCopyNum)/2)*Z[:, :, t]*np.exp(np.dot(G, H))
    # temp = Y/sumMatrix
    #
    # pool = Pool(7)
    # res = []
    # for j in range(cells):
    #     res.append(pool.apply_async(fGCOfSingleCell, (temp[:, j],)))
    # pool.close()
    # pool.join()
    # fGC = [x.get() for x in res]
    # fGC = np.array(fGC)
    # fGC[fGC <= 0] = 0.01
    # return fGC[:, :, 1].transpose()






def Estep(Y, Z, bias, pi, G, H, B, minCopyNum):
    (bins, cells, groups) = Z.shape
    likeHood = np.zeros((bins, cells, groups))
    likeHoodSum = np.zeros((bins, cells))
    biasMatrix = np.repeat(bias.reshape((bins, 1)), cells, axis=1)
    for t in range(groups):
        mu = biasMatrix*(t + minCopyNum)/2*Z[:,:,t]*np.exp(np.dot(G, H))
        piMatrix = np.repeat(pi[:, t].reshape(cells, 1), bins, axis=1).transpose()
        likeHood[:, :, t] = piMatrix*LossFunction(mu+0.001,B,Y)
        likeHoodSum += likeHood[:, :, t]

    for t in range(groups):
        Z[:, :, t] = likeHood[:, :, t]/likeHoodSum
    return Z

def iteration(originDataFrame, normalizedDataFrame, bias, K=50, alpha=1e-9, eps=5, steps=5, batchSize=100, minCopyNum=1, maxCopyNum=7, minPloidyNum = 1.5, maxPloidyNum=7):
    (bins, cells) = originDataFrame.shape
    Y = originDataFrame.values
    mu = normalizedDataFrame.values
    G = np.zeros((bins, K))
    H = np.zeros((K, cells))
    Bb = np.ones((bins, 1))
    Bc = np.ones((1, cells))
    B = np.dot(Bb, Bc)
    P = intializetionOfPloidy(Y, mu, minPloidyNum, maxPloidyNum)
    ZOld = np.zeros((bins, cells, len(list(range(minCopyNum, maxCopyNum+1)))))
    pi = np.zeros((cells, len(list(range(minCopyNum, maxCopyNum+1)))))
    biasOld = bias

    negativeControlCells, indiceOfNegativeControleCells = negativeControlCellsIdentify(originDataFrame, ginivalue=0.07)

    for i in range(bins):
        for j in range(cells):
            t = np.int(np.round(Y[i][j]*P[j]/(mu[i][j])))
            t = min(t, maxCopyNum)
            t = max(t, minCopyNum)
            ZOld[i][j][t-minCopyNum] = 1

    maxIter = 3
    iter = 0
    diffZ = float("inf")
    while iter < maxIter and diffZ > 1e-9:
        Mstep(Y, ZOld, pi, G, H, biasOld, minCopyNum)
        ZNew = Estep(Y, ZOld, biasOld, pi, G, H, B, minCopyNum)
        diffZ = diffOfTensor(ZNew, ZOld)
        print(diffZ)
        ZOld = ZNew
        iter += 1

    return ZNew


z = iteration(cellSpecificNormalizedOriginalDataFrame, cellSpecificNormalizedNorDataFrame, bias)