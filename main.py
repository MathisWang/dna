
import pandas as pd

if __name__ == '__main__':
    originFileName = "scope_experiment/B_raw.csv"
    originData = pd.read_csv(originFileName, header=0, index_col=0)
    print(originData.head())



