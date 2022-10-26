import pandas as pd
import os
import numpy as np
import removeDescriptionOfGIS
import collections
from sklearn import preprocessing

PATH = r'../数据集/continue/'

if __name__ == '__main__':
    data=np.asarray(removeDescriptionOfGIS.myread(PATH+"dem.txt"))
    v = np.asarray(removeDescriptionOfGIS.myread('./result.txt'),dtype=float).flatten()
    tag = np.asarray(np.loadtxt("./tag.csv", skiprows=1), dtype=int).flatten()
    print(max(v))
    # res = []
    #
    # count=0
    # for i in enumerate(tag):
    #     if i == 0:
    #         res.append(v[count])
    #         count+=1
    #     elif i == 1:
    #         res.append(-9999)
    # pd.DataFrame(np.asarray(res)).to_csv("res.csv", index=False, header=False)
    # matrixRecover()
