import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


originFileName = "scope_experiment/T10_raw.csv"
originData = pd.read_csv(originFileName, header=0, index_col=0)

scopeFileName = "scope_experiment/T10_nor.csv"
scopeData = pd.read_csv(scopeFileName, header=0, index_col=0)

demoFileName = "result/T10_nor_demo2.csv"
demoData = pd.read_csv(demoFileName, header=0, index_col=0)

sns_plot1 = sns.heatmap(data=originData)
plt.title("T10_raw")
plt.show()
sns_plot2 = sns.heatmap(data=scopeData)
plt.title("T10_scope_nor")
plt.show()
sns_plot3 = sns.heatmap(data=demoData)
plt.title("T10_demo_nor")
plt.show()