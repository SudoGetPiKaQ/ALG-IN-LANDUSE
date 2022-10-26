import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import precision

from pgmpy.models import BayesianModel
import numpy as np

import time
import precision
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

import sklearn

warnings.filterwarnings("ignore")
from pgmpy.estimators import HillClimbSearch, BicScore, TreeSearch

import keras
from keras import layers
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# read data


# NBN
# model_NBN = BayesianModel([(c[0], i) for i in driveFactor])
# TAN
tan = [('light_2010', 'den_school'), ('light_2010', 'den_spo'), ('light_2010', 'dist_metrosta'),
       ('light_2010', 'dist_roads'),
       ('light_2010', 'dist_railsta'), ('light_2010', 'dist_hwaysta'), ('light_2010', 'dem'),
       ('den_school', 'den_hospital'),
       ('dist_metrosta', 'dist_airpor'), ('dist_roads', 'dist_gov'),
       ('dist_railsta', 'dist_railways'), ('dem', 'dist_water'),
       ('dem', 'slope'), ('den_hospital', 'den_mall'), ('class', 'den_mall'), ('class', 'dem'),
       ('class', 'dist_roads'),
       ('class', 'dist_water'), ('class', 'dist_metrosta'), ('class', 'dist_hwaysta'),
       ('class', 'dist_railways'),
       ('class', 'slope'), ('class', 'dist_gov'), ('class', 'dist_airpor'), ('class', 'den_school'),
       ('class', 'light_2010'),
       ('class', 'den_spo'), ('class', 'dist_railsta'), ('class', 'den_hospital')]


# 服务器上运行30轮

def ann(train_data, predict_data, verifyData):
    model = keras.Sequential()
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
        optimizer='adam',
        # 注意因为label是顺序编码的,这里用这个
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )
    start = time.clock()
    model.fit(np.asarray(train_data.drop('class', axis=1, inplace=False)), np.asarray(train_data['class']),
              epochs=50, batch_size=1024,
              validation_data=(np.asarray(predict_data), np.asarray(verifyData)), verbose=0)
    y_pred = model.predict_classes(np.asarray(predict_data))
    end = time.clock()
    return end - start, y_pred, get_accuracy(y_pred, verifyData)


def rf(train_data, predict_data, verifyData):
    rf = RandomForestClassifier()
    train_drive = train_data.drop('class', inplace=False, axis=1)
    train_tag = train_data['class'].astype('int')
    start = time.clock()
    rf.fit(train_drive, train_tag)
    y_pred_rf = rf.predict(predict_data)
    end = time.clock()
    return end - start, y_pred_rf, get_accuracy(y_pred_rf, verifyData)


def lr(train_data, predict_data, verifyData):
    log_model = LogisticRegression(max_iter=10000)
    start = time.clock()
    log_model.fit(train_data.drop('class', axis=1, inplace=False), train_data['class'])
    y_pred = log_model.predict(predict_data)
    end = time.clock()
    return end - start, y_pred, get_accuracy(y_pred, verifyData)


def get_accuracy(a, b):
    return precision.clacOA(a, b), precision.f1_score(a, b, average='macro'), precision.cohen_kappa_score(a, b)


def BN(train_data, predict_data, verifyData, type):
    timeLog = {}
    edge = []
    if type == 'nbn':
        model_NBN = BayesianModel([(c[0], i) for i in driveFactor])
        start = time.clock()
        model_NBN.fit(train_data)
        y_pred_nbn = model_NBN.predict(predict_data)
        end = time.clock()
        timeLog['tan_learn'] = end - start
        return timeLog, edge, y_pred_nbn, get_accuracy(y_pred_nbn, verifyData)
    if type == 'tan':
        start = time.clock()
        edge_tan = TreeSearch(data).estimate(estimator_type="tan", class_node="class").edges()
        end = time.clock()
        timeLog['tan_learn'] = end - start
        model_TAN = BayesianModel(edge_tan)
        start = time.clock()
        model_TAN.fit(train_data)
        y_pred_tan = model_TAN.predict(predict_data)
        end = time.clock()
        timeLog['tan_infer'] = end - start
        return timeLog, edge_tan, y_pred_tan, get_accuracy(y_pred_tan, verifyData)
    if type == 'hc':
        start = time.clock()
        edge_hc = HillClimbSearch(data).estimate(scoring_method=BicScore(data)).edges()
        end = time.clock()
        timeLog['hc_learn'] = end - start
        model_HC = BayesianModel(edge_hc)
        start = time.clock()
        model_HC.fit(train_data)
        y_pred_hc = model_HC.predict(predict_data)
        end = time.clock()
        timeLog['hc_infer'] = end - start
        return timeLog, edge_hc, y_pred_hc, get_accuracy(y_pred_hc, verifyData)


all = []
for i in range(50):
    data = pd.read_csv(r'../数据集/landUse_ent.csv', index_col=False)
    data0 = data.loc[data['class'] == 0].sample(frac=0.05)
    data1 = data.loc[data['class'] == 1]
    data2 = data.loc[data['class'] == 2]
    print(len(data0), len(data1), len(data2))
    data = pd.concat([data0, data1, data2], ignore_index=True)
    driveFactor = data.columns.to_list()[:-1]
    c = data.columns.to_list()[-1:]
    train_data = data.sample(frac=0.7)
    test_data = data[~data.index.isin(train_data.index)]
    verifyData = test_data['class']
    predict_data = test_data.copy()
    predict_data.drop('class', axis=1, inplace=True)
    log = {"edge": {}, 'oa': [], 'f1': [], 'kappa': []}
    verifyData.to_csv(f'./verify/verify_{i}.csv', index=False, header=False)
    try:
        # NBN
        timeLog, edge_nbn, y_pred_nbn, accuracy = BN(train_data, predict_data, verifyData, 'nbn')
        log['nbn_time'] = timeLog
        log['oa'].append(accuracy[0])
        log['f1'].append(accuracy[1])
        log['kappa'].append(accuracy[2])
        y_pred_nbn.to_csv(f'./pred/nbn/NBN_y_pred_{i}.csv', index=False, header=False)
        # TAN
        timeLog, edge_tan, y_pred_tan, accuracy = BN(train_data, predict_data, verifyData, 'tan')
        log['oa'].append(accuracy[0])
        log['f1'].append(accuracy[1])
        log['kappa'].append(accuracy[2])
        log['edge']['tan'] = edge_tan
        log['tan_time'] = timeLog
        y_pred_tan.to_csv(f'./pred/tan/TAN_y_pred_{i}.csv', index=False, header=False)
        # HC
        timeLog, edge_hc, y_pred_hc, accuracy = BN(train_data, predict_data, verifyData, 'hc')
        log['hc_time'] = timeLog
        log['edge']['hc'] = edge_hc
        log['oa'].append(accuracy[0])
        log['f1'].append(accuracy[1])
        log['kappa'].append(accuracy[2])
        y_pred_hc.to_csv(f'./pred/hc/HC_y_pred_{i}.csv', index=False, header=False)

        # logistic
        timeLog, y_pred_lr, accuracy = lr(train_data, predict_data, verifyData)
        log['oa'].append(accuracy[0])
        log['f1'].append(accuracy[1])
        log['kappa'].append(accuracy[2])
        log['logistic_time'] = timeLog
        pd.DataFrame(y_pred_lr).to_csv(f'./pred/lr/LR_y_pred_{i}.csv', index=False, header=False)

        # RF
        timeLog, y_pred_rf, accuracy = rf(train_data, predict_data, verifyData)
        log['oa'].append(accuracy[0])
        log['f1'].append(accuracy[1])
        log['kappa'].append(accuracy[2])
        log['rf_time'] = timeLog
        pd.DataFrame(y_pred_rf).to_csv(f'./pred/rf/RF_y_pred_{i}.csv', index=False, header=False)

        # ANN
        timeLog, y_pred_ann, accuracy = ann(train_data, predict_data, verifyData)
        log['oa'].append(accuracy[0])
        log['f1'].append(accuracy[1])
        log['kappa'].append(accuracy[2])
        log['ann_time'] = timeLog
        pd.DataFrame(y_pred_ann).to_csv(f'./pred/ann/ANN_y_pred_{i}.csv', index=False, header=False)

    except Exception as e:
        print(e.args)
        break
    finally:
        all.append(log)
        print(log)
        with open(f'./log/log.txt', mode='a+') as edge:
            edge.write(str(log))

# infer = VariableElimination(model)
# print(infer.query(['class']))
# print(infer.query(variables=['class'], evidence={'jenk_spot_10_': 1}))

# for c in model.get_cpds():
#     print(c)

# for i in model.get_cpds():
#     print(i)


# print(precision.clac(y_pred.values,verifyData.values))
# isItRightOrNot = y_pred.values.flatten() - verifyData.values.flatten() == 0
# pd.DataFrame(np.hstack((np.asarray(test_data), np.asarray(y_pred, dtype=int).reshape(-1, 1),
#                  np.asarray(isItRightOrNot, dtype=int).reshape(-1, 1)))).to_excel('预测结果.xlsx')

# trueCount = sum(y_pred.values.flatten() - verifyData.values.flatten() == 0) / len(
#     y_pred.values.flatten() - verifyData.values.flatten() == 0)


# res=np.asarray(y_pred).flatten()
# print('--------------------total--------------------')
# print(sum(res==verifyData.values)/verifyData.values.shape[0])
#
# print('----------------------1-----------------')
# index=np.where(np.asarray(verifyData.values)==1)
# print(sum(res[index]==1) / len(index[0]))
#
# print('----------------------0-----------------')
#
# index=np.where(np.asarray(verifyData.values)==0)
# print(sum(res[index]==0) / len(index[0]))
#
# print('----------------------2-----------------')
# index=np.where(np.asarray(verifyData.values)==2)
# print(sum(res[index]==2) / len(index[0]))


'''
HC
'''
# hc = HillClimbSearch(data)
# best_model = hc.estimate(scoring_method=BicScore(data))
# print(best_model.edges())

'''
TAN
'''
# est = TreeSearch(data)
# G = est.estimate(estimator_type="tan", class_node="class")
# G.remove_node("class")


'''
MMHC
'''
# est = MmhcEstimator(data)
# model = est.estimate(significance_level=0.3)
# print(model.edges())


'''
pos = nx.spring_layout(G)  # positions for all nodes
print(G.edges())
# nodes
options = {"node_size": 500, "alpha": 0.8}
# nx.draw_networkx_nodes(G, pos, nodelist=["class"], node_color="r", **options)
nx.draw_networkx_nodes(G, pos, nodelist=["light_2010"], node_color="b", **options)
nx.draw_networkx_nodes(G, pos, nodelist=driveFactor, **options)


# edges
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[],
    width=8,
    alpha=0.5,
    edge_color="r",
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[],
    width=8,
    alpha=0.5,
    edge_color="b",
)


# some math labels
nx.draw_networkx_labels(G, pos, font_size=16)
plt.axis("off")
plt.show()
# nx.draw_circular(dag, , arrowsize=10, node_size=800, alpha=0.8, font_weight='bold',font_size=8)

'''
