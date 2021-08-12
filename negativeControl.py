import pandas as pd
import numpy as np

originFileName = "scope_experiment/B_raw.csv"
originData = pd.read_csv(originFileName, header=0, index_col=0)


def gini(x):
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def negativeControlCellsIdentify(dataFrame, ginivalue=0.08):
    binsNames = dataFrame._stat_axis.values.tolist()
    cellsNames = dataFrame.columns.values.tolist()
    (bins, cells) = dataFrame.shape
    data = dataFrame.values
    indiceOfNegativeControlCells = []
    negativeControlCellsNames = []
    negativeControlCellsValues = []


    for c in range(cells):
        if gini(data[:, c]) < ginivalue:
            indiceOfNegativeControlCells.append(c)
            negativeControlCellsNames.append(cellsNames[c])
            negativeControlCellsValues.append(data[:, c])

    negativeControlCellsValues = np.array(negativeControlCellsValues)
    negativeControlCellsValues = negativeControlCellsValues.transpose()
    negativeControlCellsData = pd.DataFrame(negativeControlCellsValues, index=binsNames, columns=negativeControlCellsNames)
    return negativeControlCellsData, indiceOfNegativeControlCells


