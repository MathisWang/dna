import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error as mse
from scipy.special import gammaln
from numpy import log
import random
from numba.extending import get_cython_function_address
from numba import vectorize, njit, prange, jit
import numba as nb
import ctypes


_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
addr1 = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1psi")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)
psi_float64 = functype(addr1)



@jit(nopython=True)
def numba_psi(x):
    return psi_float64(x)

@jit(nopython=True)
def numba_gammaln(x):
  return gammaln_float64(x)



@jit(nopython=True, parallel=True)
def gammaln_array2D(x):
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            x[i][j] = numba_gammaln(x[i][j])
    return x



def dataCellSpecificLibrarySizeFactors(dataFrame):
    """
    calculate cell-specific library size factor
    :param dataFrame: 输入的dataframe
    :return: 标准化后的数据，以及每个细胞对应的sizefactor
    """
    librarySize = dataFrame.sum(axis=0)
    meanLibrarySize = librarySize.mean()
    sizeFactors = librarySize / meanLibrarySize
    cellSpecificNormalizedDataFrame = dataFrame / sizeFactors
    return cellSpecificNormalizedDataFrame, sizeFactors


def dataBinSpecificBias(negativeCellsDataFrame, indiceOfNegativeControlCells, G, H):
    matrix = np.exp(np.dot(G, H))
    biasEstimatedMatrix = negativeCellsDataFrame.values/(matrix[:, indiceOfNegativeControlCells])
    biasNew = np.median(biasEstimatedMatrix, axis=1)
    return biasNew


def initialMatrice(decomposedMatrix, nFeatures):
    svd = TruncatedSVD(n_components=nFeatures)
    svd.fit(decomposedMatrix)
    G = svd.fit_transform(decomposedMatrix)
    H = svd.components_
    return G, H


@jit(nb.float64[:, :](nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]),
     nopython=True, fastmath=True, parallel=True)
def LossFunction(mu, b, y):
    """

    :param mu:
    :param b:
    :param y:
    :return:
    """
    fun1 = - (mu / b) * log(b)
    fun2 = -gammaln_array2D(mu / b)
    fun3 = gammaln_array2D(y + (mu / b))
    fun4 = -(y + (mu / b)) * log(1 + (1 / b))
    fun = fun1 + fun2 + fun3 + fun4

    (rows, cols) = fun.shape
    fun = np.dot(fun, np.ones((cols, 1)))
    fun = np.dot(fun.transpose(), np.ones((rows, 1)))
    return fun


# def LossFunction(mu, b, y):
#     fun1 = - (mu / b) * log(b)
#     fun2 = -gammaln(mu / b)
#     fun3 = gammaln(y + (mu / b))
#     fun4 = -(y + (mu / b)) * log(1 + (1 / b))
#     fun = fun1 + fun2 + fun3 + fun4
#
#     (rows, cols) = fun.shape
#     fun = np.dot(fun, np.ones((cols, 1)))
#     fun = np.dot(fun.transpose(), np.ones((rows, 1)))
#
#     return np.float64(fun)

def chooseNegativeControlCells(indicesOfNegativeControlCells, batchSize):
    if len(indicesOfNegativeControlCells) >= batchSize:
        negativeControlCellsChosen = random.sample(indicesOfNegativeControlCells, batchSize)
    else:
        negativeControlCellsChosenFrom = np.repeat(indicesOfNegativeControlCells, int(batchSize/len(indicesOfNegativeControlCells))+1)
        negativeControlCellsChosen = random.sample(list(negativeControlCellsChosenFrom), batchSize)
    return negativeControlCellsChosen


def diffOfTensor(tensor1, tensor2):
    (length, width, depth) = tensor1.shape
    res = 0
    for d in range(depth):
        res += mse(tensor1[:, :, d], tensor2[:, :, d])
    return res

def diffOfBias(biasOld, biasNew):
    res = mse(biasOld, biasNew)
    return res



