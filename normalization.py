import pandas as pd
import numpy as np
from utiles import dataCellSpecificLibrarySizeFactors, dataBinSpecificBias, initialMatrice, LossFunction, chooseNegativeControlCells, diffOfBias
from negativeControl import negativeControlCellsIdentify
from optimize import optimize
import matplotlib.pyplot as plt
from dpSegmentation import dpKmeans


originFileName = "scope_experiment/T16P_raw.csv"
originData = pd.read_csv(originFileName, header=0, index_col=0)
binsNames = originData._stat_axis.values.tolist()
cellsNames = originData.columns.values.tolist()
cellSpecificNormalizedDataFrame, sizeFactors = dataCellSpecificLibrarySizeFactors(originData)
print(cellSpecificNormalizedDataFrame)

def normalization(dataFrame, K, alpha, eps, steps, iteratonNum, batchSize):
    (bins, cells) = dataFrame.shape
    bias0 = np.ones((1, bins))
    biasOld = bias0
    G = np.zeros((bins, K))+1e-1
    H = np.zeros((K, cells))+1e-1
    Bb = np.zeros((bins, 1))
    Bc = np.zeros((1, cells))
    negativeControlCells, indiceOfNegativeControleCells = negativeControlCellsIdentify(dataFrame, ginivalue=0.12)
    Y = dataFrame.values
    negativeControlCellsNum = negativeControlCells.shape[1]


    for step in range(steps):
        biasNew = dataBinSpecificBias(negativeControlCells, indiceOfNegativeControleCells, G, H)
        if diffOfBias(biasOld, biasNew.reshape(1,bins)) <= 1e-5:
            break
        else:
            print("diffBias", diffOfBias(biasOld, biasNew.reshape(1,bins)))
            biasNewMatrix = np.repeat(biasNew.reshape((bins, 1)), cells, axis=1)
            biasOld = biasNew.reshape(1,bins)

            a = np.log(Y+0.01/biasNewMatrix)
            G, H = initialMatrice(a, nFeatures=K)

            lossFuncs = []
            lossFuncPre = float("-inf")

            for i in range(iteratonNum):
                mu = biasNewMatrix*np.exp(np.dot(G, H))
                mu[mu <= 0] = 0
                B = np.dot(Bb, Bc)
                B[B <= 0] = 1e-10
                lossFunc = np.float64(LossFunction(mu, B, Y))

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
            print(step+1)
            plt.plot(lossFuncs)
            plt.show()

    return biasNew, mu, B, G, H


bias, mu, B, leftMetrix, rightMetrix = normalization(cellSpecificNormalizedDataFrame, K=30, alpha=1e-10, eps=1, steps=10, iteratonNum=20, batchSize=100)



normalizedResult = pd.DataFrame(mu, index=binsNames, columns=cellsNames)
normalizedResult = normalizedResult*sizeFactors
intergerMu = normalizedResult.values.astype(np.int)
normalizedResult = pd.DataFrame(intergerMu, index=binsNames, columns=cellsNames)
normalizedResult.to_csv("result/T16P_nor_demo.csv")
biasDataFrame = pd.DataFrame(bias)
biasDataFrame.to_csv("result/T16P_ref_demo.csv")

leftMatrixDataFrame = pd.DataFrame(leftMetrix, index=binsNames, columns=None)
rightMatrixDataFrame = pd.DataFrame(rightMetrix, index=None, columns=cellsNames)

leftMatrixDataFrame.to_csv("MFResult/T16P_leftMatrix_Fano.csv")
rightMatrixDataFrame.to_csv("MFResult/T16P_rightMatrix_Fano.csv")



























