import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,cohen_kappa_score

def clac3(a1,a2):
    a1=np.asarray(a1).flatten()
    a2=np.asarray(a2).flatten()
    arr=np.empty([3,3])
    arr[0][0]=np.sum(np.logical_and(a1==0,a2==0))
    arr[0][1]=np.sum(np.logical_and(a1==0,a2==1))
    arr[0][2]=np.sum(np.logical_and(a1==0,a2==2))
    arr[1][0]=np.sum(np.logical_and(a1==1,a2==0))
    arr[1][1]=np.sum(np.logical_and(a1==1,a2==1))
    arr[1][2]=np.sum(np.logical_and(a1==1,a2==2))
    arr[2][0]=np.sum(np.logical_and(a1==2,a2==0))
    arr[2][1]=np.sum(np.logical_and(a1==2,a2==1))
    arr[2][2]=np.sum(np.logical_and(a1==2,a2==2))
    predict0=np.sum(a1==0)
    predict1=np.sum(a1==1)
    predict2=np.sum(a1==2)
    true0=np.sum(a2==0)
    true1=np.sum(a2==1)
    true2=np.sum(a2==2)
    totalTrue=np.sum(np.diagonal(arr))
    OA= totalTrue / len(a1)
    pe=(predict0*true0+predict1*true1+predict2*true2)/len(a1)**2
    Kappa=(OA-pe)/(1-pe)
    return OA,Kappa

def clac2(a1,a2):
    a1 = np.asarray(a1).flatten()
    a2 = np.asarray(a2).flatten()
    arr = np.empty([2, 2])
    arr[0][0] = np.sum(np.logical_and(a1 == 0, a2 == 0))
    arr[0][1] = np.sum(np.logical_and(a1 == 0, a2 == 1))
    arr[1][0] = np.sum(np.logical_and(a1 == 1, a2 == 0))
    arr[1][1] = np.sum(np.logical_and(a1 == 1, a2 == 1))
    predict0 = np.sum(a1 == 0)
    predict1 = np.sum(a1 == 1)
    true0 = np.sum(a2 == 0)
    true1 = np.sum(a2 == 1)
    totalTrue = np.sum(np.diagonal(arr))
    OA = totalTrue / len(a1)
    pe = (predict0 * true0 + predict1 * true1) / len(a1) ** 2
    Kappa = (OA - pe) / (1 - pe)
    return OA, Kappa

def clacOA(a1,a2):
    a1 = np.asarray(a1).flatten()
    a2 = np.asarray(a2).flatten()
    return np.sum(a1==a2)/len(a1)

if __name__ == '__main__':
    v=np.asarray(pd.read_csv('./verify.csv'))
    p=np.asarray(pd.read_csv('./y_pred.csv'))
    print(clacOA(p,v),f1_score(p, v,average='macro'),cohen_kappa_score(p, v))