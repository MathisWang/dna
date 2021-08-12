import pandas as pd
import numpy as np
from utiles import dataCellSpecificLibrarySizeFactors, dataBinSpecificBias, initialMatrice, LossFunction, chooseNegativeControlCells, diffOfTensor, diffOfBias
from negativeControl import negativeControlCellsIdentify
from optimize import optimize
from multiprocessing import Pool
import statsmodels.api as sm



originFileName = "scope_experiment/T10_raw.csv"
originData = pd.read_csv(originFileName, header=0, index_col=0)
binsNames = originData._stat_axis.values.tolist()
cellsNames = originData.columns.values.tolist()
cellSpecificNormalizedOriginalDataFrame, sizeFactors = dataCellSpecificLibrarySizeFactors(originData)


normalizedFileName = "result/T10_nor_demo2.csv"
normalizedData = pd.read_csv(normalizedFileName, header=0, index_col=0)
cellSpecificNormalizedNorDataFrame, sizeFactors1 = dataCellSpecificLibrarySizeFactors(normalizedData)

biasFileName = "result/T10_ref_demo2.csv"
biasData = pd.read_csv(biasFileName, header=0, index_col=0)

fGCFileName = "result/T10_fGC_demo2.csv"
fGCData = pd.read_csv(fGCFileName, header=0, index_col=0)


Y = cellSpecificNormalizedOriginalDataFrame.values
mu = cellSpecificNormalizedNorDataFrame.values
bias = biasData.values
fGC = fGCData.values

# def absoluteCopyNumber(Y, mu, p):
#     r = Y*p/mu
#     rInt = round(r)
#     res = sum((r-rInt)**2)
#     return res

def intializetionOfPloidy(Y, mu, minPloidyNum, maxPloidyNum):
    mu = mu + 0.01
    res = []
    (bins, cells) = Y.shape
    # P = list(range(minPloidyNum, maxPloidyNum+1,0.05))
    P = np.linspace(minPloidyNum, maxPloidyNum, num=(np.int((maxPloidyNum-minPloidyNum)/0.05)))
    for j in range(cells):
        diff = list(map(lambda x: sum(((Y[:, j]*x/mu[:, j])-(np.round(Y[:, j]*x/mu[:, j])))**2), P))
        p = P[np.argmin(diff)]
        res.append(p)
    return np.array(res)

def fGCOfSingleCell(temp):
    lowess = sm.nonparametric.lowess
    fgc = lowess(temp, list(range(len(temp))), frac=0.45)
    return fgc


def Mstep(Y, Z, pi, G, H, bias, minCopyNum):
    (bins, cells, groups) = Z.shape
    for j in range(cells):
        for t in range(groups):
            pi[j][t] = sum(Z[:, j, t])/bins


    sumMatrix = np.zeros((bins, cells))
    biasNewMatrix = np.repeat(bias.reshape((bins, 1)), cells, axis=1)
    for t in range(groups):
        sumMatrix += biasNewMatrix*((t+minCopyNum)/2)*Z[:, :, t]*np.exp(np.dot(G, H))
    temp = Y/sumMatrix

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






def Estep(Y, Z, bias, fGC, pi, G, H, B, minCopyNum):
    (bins, cells, groups) = Z.shape
    ZNew = np.zeros(Z.shape)
    expOfCopyNum = np.zeros((bins, cells))
    likeHood = np.zeros((bins, cells, groups))
    likeHoodSum = np.zeros((bins, cells))
    biasMatrix = np.repeat(bias.reshape((bins, 1)), cells, axis=1)
    for t in range(groups):
        mu = biasMatrix*fGC*(t + minCopyNum)/2*Z[:,:,t]*np.exp(np.dot(G, H))
        piMatrix = np.repeat(pi[:, t].reshape(cells, 1), bins, axis=1).transpose()
        likeHood[:, :, t] = piMatrix*LossFunction(mu+0.001,B,Y)
        likeHoodSum += likeHood[:, :, t]

    for t in range(groups):
        ZNew[:, :, t] = likeHood[:, :, t]/likeHoodSum
        expOfCopyNum += ((t + minCopyNum)/2)*ZNew[:, :, t]
    return ZNew, expOfCopyNum





def iteration(originDataFrame, normalizedDataFrame, bias,fGC, K, alpha, eps, steps, batchSize, minCopyNum=1, maxCopyNum=7):
    (bins, cells) = originDataFrame.shape
    Y = originDataFrame.values
    mu = normalizedDataFrame.values
    G = np.zeros((bins, K))
    H = np.zeros((K, cells))
    Bb = np.ones((bins, 1))
    Bc = np.ones((1, cells))
    B = np.dot(Bb, Bc)
    P = intializetionOfPloidy(Y, mu, minPloidyNum=1.5, maxPloidyNum=7)
    ZOld = np.zeros((bins, cells, len(list(range(minCopyNum, maxCopyNum+1)))))
    pi = np.zeros((cells, len(list(range(minCopyNum, maxCopyNum+1)))))
    biasOld = bias
    negativeControlCells, indiceOfNegativeControleCells = negativeControlCellsIdentify(originDataFrame, ginivalue=0.1)

    for i in range(bins):
        for j in range(cells):
            t = np.int(np.round(Y[i][j]*P[j]/(biasOld[i]*fGC[i][j])))
            t = min(t, maxCopyNum)
            t = max(t, minCopyNum)
            ZOld[i][j][t-minCopyNum] = 1

    for step in range(steps):
        maxIter = 3
        iter = 0
        diffZ = float("inf")
        while iter < maxIter and diffZ > 1e-3:
            fGC = Mstep(Y, ZOld, pi, G, H, biasOld, minCopyNum)
            ZNew, expOfCopyNum = Estep(Y, ZOld, biasOld, fGC, pi, G, H, B, minCopyNum)
            diffZ = diffOfTensor(ZOld, ZNew)
            print(diffZ)
            print(iter)
            ZOld = ZNew
            iter += 1

        biasNew = dataBinSpecificBias(negativeControlCells, indiceOfNegativeControleCells, G, H, fGC)

        if diffOfBias(biasOld, biasNew.reshape(bins, 1)) <= 1e-4:
            break
        else:
            print("diffBias", diffOfBias(biasOld, biasNew.reshape(bins, 1)))
            biasNewMatrix = np.repeat(biasNew.reshape((bins, 1)), cells, axis=1)
            biasOld = biasNew.reshape(bins, 1)


            a = np.log((Y + 0.01) / (biasNewMatrix * fGC * expOfCopyNum))
            G, H = initialMatrice(a, nFeatures=K)
            lossFuncs = []
            lossFuncPre = float("-inf")

            for it in range(5):
                mu = fGC * biasNewMatrix * expOfCopyNum * np.exp(np.dot(G, H))
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
                    CellsChosen = np.random.randint(low=0, high=cells, size=batchSize)
                    for i, j in zip(binsChosen, CellsChosen):
                        dG, dH, dBb, dBc = optimize(mu, B, Y, G, H, Bb, Bc, i, j)
                        G[i, :] = G[i, :] + alpha * dG
                        H[:, j] = H[:, j] + alpha * dH
                        Bb[i, :] = Bb[i, :] + alpha * dBb
                        Bc[:, j] = Bc[:, j] + alpha * dBc
    return mu, ZNew, biasNew, B, fGC







if __name__ == "__main__":
    mu, z, bias, B, f = iteration(cellSpecificNormalizedOriginalDataFrame,cellSpecificNormalizedNorDataFrame,bias,fGC,50,1e-9,1,3,60)
    segmentation = np.zeros(mu.shape)
    (bins, cells, groups) = z.shape
    for i in range(bins):
        for j in range(cells):
            segmentation[i][j] = np.argmax(z[i, j, :]) + 1

    print(segmentation)


























