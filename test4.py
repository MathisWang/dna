# _*_ coding: utf-8 _*_
"""
Time:     2021/8/11 18:49
Author:   WANG Bingchen
Version:  V 0.1
File:     test4.py
Describe: 
"""

import pandas as pd
import numpy as np
from utiles import dataCellSpecificLibrarySizeFactors, dataBinSpecificBias, initialMatrice, LossFunction, chooseNegativeControlCells, diffOfBias
from negativeControl import negativeControlCellsIdentify
from optimize import optimize
import matplotlib.pyplot as plt
from dpSegmentation import dpKmeans
import warnings
warnings.filterwarnings("ignore")

originFileNameP = "scope_experiment/T16P_raw.csv"
originDataP = pd.read_csv(originFileNameP, header=0, index_col=0)
originDataP = originDataP.iloc[:, 0:48]




originFileNameM = "scope_experiment/T16M_raw.csv"
originDataM = pd.read_csv(originFileNameM, header=0, index_col=0)



common_index = [i for i in originDataP.index if i in originDataM.index]

originDataM = originDataM.loc[common_index]
originDataP = originDataP.loc[common_index]

print(originDataM)
print(originDataP)



originData = pd.DataFrame(originDataM.values + originDataP.values, index= common_index)


print(originData)



binsNames = originData._stat_axis.values.tolist()
cellsNames = originData.columns.values.tolist()
cellSpecificNormalizedDataFrame, sizeFactors = dataCellSpecificLibrarySizeFactors(originData)
print(cellSpecificNormalizedDataFrame)


def normalization(dataFrame, K, alpha, eps, steps, iteratonNum, batchSize):
    (bins, cells) = dataFrame.shape
    bias0 = np.ones((1, bins))
    biasOld = bias0
    G = np.zeros((bins, K, 2))+1e-1
    H = np.zeros((K, cells, 2))+1e-1
    Bb = np.zeros((bins, 1, 2))
    Bc = np.zeros((1, cells, 2))
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

            a = np.log(Y/2+0.01/biasNewMatrix)
            G[:, :, 0], H[:, :, 0] = initialMatrice(a, nFeatures=K)
            G[:, :, 1], H[:, :, 1] = initialMatrice(a, nFeatures=K)


            lossFuncs = []
            lossFuncPre = float("-inf")

            for i in range(iteratonNum):
                mu = biasNewMatrix*np.exp(np.dot(G[:, :, 0], H[:, :, 0])) + biasNewMatrix*np.exp(np.dot(G[:, :, 1], H[:, :, 1]))
                mu[mu <= 0] = 0
                B = np.dot(Bb[:, :, 0], Bc[:, :, 0])
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
                        dG, dH, dBb, dBc = optimize(mu, B, Y, G[:, :, 0], H[:, :, 0], Bb[:, :, 0], Bc[:, :, 0], i, j)
                        G[i, :, 0] = G[i, :, 0] + alpha*dG
                        H[:, j, 0] = H[:, j, 0] + alpha*dH
                        Bb[i, :, 0] = Bb[i, :, 0] + alpha*dBb
                        Bc[:, j, 0] = Bc[:, j, 0] + alpha*dBc

                    binsChosen = np.random.randint(low=0, high=bins, size=batchSize)
                    negativeControlCellsChosen = chooseNegativeControlCells(indiceOfNegativeControleCells, batchSize)
                    for i, j in zip(binsChosen, negativeControlCellsChosen):
                        dG, dH, dBb, dBc = optimize(mu, B, Y, G[:, :, 1], H[:, :, 1], Bb[:, :, 1], Bc[:, :, 1], i, j)
                        G[i, :, 1] = G[i, :, 1] + alpha * dG
                        H[:, j, 1] = H[:, j, 1] + alpha * dH
                        Bb[i, :, 1] = Bb[i, :, 1] + alpha * dBb
                        Bc[:, j, 1] = Bc[:, j, 1] + alpha * dBc
            print(step+1)
            plt.plot(lossFuncs)
            plt.show()

    return biasNew, mu, B, G, H


bias, mu, B, leftMetrix, rightMetrix = normalization(cellSpecificNormalizedDataFrame, K=30, alpha=1e-11, eps=1, steps=10, iteratonNum=20, batchSize=100)



normalizedResult = pd.DataFrame(mu, index=binsNames, columns=cellsNames)
normalizedResult = normalizedResult*sizeFactors
intergerMu = normalizedResult.values.astype(np.int)
normalizedResult = pd.DataFrame(intergerMu, index=binsNames, columns=cellsNames)
# normalizedResult.to_csv("2phase/T16_nor_demo.csv")
# biasDataFrame = pd.DataFrame(bias)
# biasDataFrame.to_csv("2phase/T16_ref_demo.csv")
#
# leftMatrixDataFrame = pd.DataFrame(leftMetrix, index=binsNames, columns=None)
# rightMatrixDataFrame = pd.DataFrame(rightMetrix, index=None, columns=cellsNames)
#
# leftMatrixDataFrame.to_csv("2phase/T16_leftMatrix_`Fano.csv")
# rightMatrixDataFrame.to_csv("2phase/T16_rightMatrix_Fano.csv")