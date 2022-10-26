import pandas as pd
import os
import numpy as np
import removeDescriptionOfGIS
import collections
from sklearn import preprocessing
PATH='../数据集/continue/'
li=os.listdir(PATH)


'''
col=[i for i in list]
col.append('class')
print(col)
'''

def removeBlankAndToVector(arr:np.ndarray)->np.ndarray:
    return np.delete(arr.flatten(), np.where(arr.flatten() == -9999)).reshape(-1, 1)




def matrixRemoveBlank(arr):
    res=[]
    tag=[]
    for i in arr:
        if sum(i==-9999)>0:
            tag.append(1)
            continue
        else:
            tag.append(0)
            res.append(i)
    return np.asarray(res),np.asarray(tag).reshape(-1,1)

landUseData=np.asarray(np.loadtxt(r'../数据集/data/urban_change_1017.txt', skiprows=6),dtype=int).flatten().reshape(-1,1)

columns=collections.deque(maxlen=20)

t=0

for i in li:
    print(i)
    # data=removeBlankAndToVector(np.asarray(pd.read_table(PATH + i, header=None, delim_whitespace=True, index_col=None, encoding='gb2312',dtype=int)))
    data=np.asarray(removeDescriptionOfGIS.myread(PATH+i),dtype=float).flatten().reshape(-1,1)
    t+=1
    data[0]=t
    print(t)
    '''
    0.0,331.0,51145.38,7566.373,89226.06,9902.02,11600.0,3.15134,36537.79,112918.8,0.0,0.0,0.008518772,24203.3,0.0,0.0
    '''
    landUseData=np.hstack((data,landUseData))
    columns.appendleft(i.rstrip('.txt'))
data,tag=matrixRemoveBlank(landUseData)

# scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
# scaler.fit(data)
# data=scaler.transform(data)
# data=np.round(data,decimals=6)

# save
columns.append('class')
pd.DataFrame(tag).to_csv('tag.csv', index=False)
pd.DataFrame(data).to_csv('landUse_src.csv', index=False, header=list(columns))
