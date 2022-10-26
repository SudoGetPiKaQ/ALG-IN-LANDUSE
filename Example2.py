import pandas as pd
from pgmpy.models import BayesianModel
import networkx as nx
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch, BicScore

# read data
NUM = 21
data = pd.read_csv('./数据集/landUse.csv')
driveFactor = data.columns.to_list()[:-1]
c = data.columns.to_list()[-1:]


def getAccuracy(model, data, verifyData):
    predict_data = data.copy()
    predict_data.drop('type', axis=1, inplace=True)
    y_pred = model.predict(predict_data)
    trueCount = sum(y_pred.values.flatten() - verifyData.values.flatten() == 0) / len(
        y_pred.values.flatten() - verifyData.values.flatten() == 0)
    return trueCount


# get best model
bestModel = 0
bestAccuracy = 0
verifyData = data['type']
driveData = data
for i in range(1, 100):
    model = BayesianModel([(c[0], i) for i in driveFactor])
    train_data = data.sample(frac=0.7)
    model.fit(train_data)
    if getAccuracy(model, driveData, verifyData) > bestAccuracy:
        bestModel = model
        bestAccuracy = getAccuracy(model, driveData, verifyData)

# get which true/wrong
newData = data.copy()
verifyData = newData['type']
driveData = newData.drop('type', axis=1, inplace=False)
y_pred = bestModel.predict(driveData)
isItRightOrNot = y_pred.values.flatten() - verifyData.values.flatten() == 0
pd.concat([data,
    pd.DataFrame(np.hstack((np.asarray(y_pred).reshape(-1, 1), np.asarray(isItRightOrNot).reshape(-1, 1))),
                 columns=['predict', 'isTrue'])],axis=1).to_csv('预测结果.csv')

# draw
# saveAccuracy = []
# for i in range(1, NUM):
#     test_data = data.sample(frac=0.3)
#     verifyData = test_data['accident_type']
#     driveData = test_data
#     saveAccuracy.append(getAccuracy(bestModel, driveData, verifyData))
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# fig = plt.figure(figsize=(10,9),dpi=300)
# ax = fig.add_subplot(1, 1, 1)
# ax.plot([i for i in range(1,NUM)], saveAccuracy, 'b')
# ax.set_ylim([0, 1])
# ax.set_title('基于贝叶斯网络的预测准确率')
# ax.set_xlabel('训练次数(次)')
# ax.set_ylabel('准确率(%)')
# plt.savefig('./img2.png')
