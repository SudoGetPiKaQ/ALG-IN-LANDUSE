import numpy as np
def myread(filename):
    #去除arcgis导出txt文件的前6行数据，方便在numpy中处理
    data = np.loadtxt(rf'{filename}', skiprows=0)
    return data
