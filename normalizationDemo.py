import pandas as pd
import numpy as np
from utiles import dataCellSpecificLibrarySizeFactors, dataBinSpecificBias, initialMatrice, LossFunction, chooseNegativeControlCells, diffOfTensor, diffOfBias
from negativeControl import negativeControlCellsIdentify
from optimize import optimize
import statsmodels.api as sm
from multiprocessing import Pool
import matplotlib.pyplot as plt






def fGCOfSingleCell(temp):
    lowess = sm.nonparametric.lowess
    fgc = lowess(temp, list(range(len(temp))))
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


def normalization(dataFrame, K, alpha, eps, steps, iteratonNum, batchSize):
    (bins, cells) = dataFrame.shape
    bias0 = np.ones((1, bins))
    biasOld = bias0
    G = np.zeros((bins, K))
    H = np.zeros((K, cells))
    Bb = np.ones((bins, 1))
    Bc = np.ones((1, cells))
    negativeControlCells, indiceOfNegativeControleCells = negativeControlCellsIdentify(dataFrame, ginivalue=0.1)
    Y = dataFrame.values

    for step in range(steps):
        print("step", step+1)
        fGC = getFGC(Y, bins, cells, biasOld, G, H)
        biasNew = dataBinSpecificBias(negativeControlCells, indiceOfNegativeControleCells, G, H, fGC)

        if diffOfBias(biasOld, biasNew.reshape(1,bins)) <= 1e-5:
            break
        else:
            print("diffBias",diffOfBias(biasOld, biasNew.reshape(1,bins)))
            biasNewMatrix = np.repeat(biasNew.reshape((bins, 1)), cells, axis=1)
            biasOld = biasNew.reshape(1,bins)

            a = np.log((Y + 0.01) / (biasNewMatrix*fGC))
            G, H = initialMatrice(a, nFeatures=K)

            lossFuncs = []
            lossFuncPre = float("-inf")

            for iter in range(iteratonNum):
                mu = fGC*biasNewMatrix * np.exp(np.dot(G, H))
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
                        G[i, :] = G[i, :] + alpha * dG
                        H[:, j] = H[:, j] + alpha * dH
                        Bb[i, :] = Bb[i, :] + alpha * dBb
                        Bc[:, j] = Bc[:, j] + alpha * dBc
            print("iter",iter + 1)
            plt.plot(lossFuncs)
            plt.show()
    return mu, biasNew, B, fGC


if __name__ == "__main__":
    originFileName = "scope_experiment/T10_raw.csv"
    originData = pd.read_csv(originFileName, header=0, index_col=0)
    binsNames = originData._stat_axis.values.tolist()
    cellsNames = originData.columns.values.tolist()
    cellSpecificNormalizedOriginalDataFrame, sizeFactors = dataCellSpecificLibrarySizeFactors(originData)



    pred, bias, B, fGC = normalization(cellSpecificNormalizedOriginalDataFrame, K=50, alpha=1e-10, eps=1, steps=5, iteratonNum=10,
                                batchSize=100)
    normalizedResult = pd.DataFrame(pred, index=binsNames, columns=cellsNames)
    normalizedResult = normalizedResult * sizeFactors
    intergerMu = normalizedResult.values.astype(np.int)
    normalizedResult = pd.DataFrame(intergerMu, index=binsNames, columns=cellsNames)
    print(normalizedResult)
    normalizedResult.to_csv("result/T10_nor_demo2.csv")
    biasDataFrame = pd.DataFrame(bias)
    biasDataFrame.to_csv("result/T10_ref_demo2.csv")
    fGCDataFrame = pd.DataFrame(fGC)
    fGCDataFrame.to_csv("result/T10_fGC_demo2.csv")



