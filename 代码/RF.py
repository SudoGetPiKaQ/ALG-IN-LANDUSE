from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import sklearn
import time

rf = RandomForestClassifier()  # 这里使用了默认的参数设置
# data = pd.read_csv('../数据集/landUse_continue.csv',index_col=False)
data = pd.read_csv('../数据集/landUse_normalize.csv', index_col=False)

data0 = data.loc[data['class'] == 0].sample(frac=0.05)
data1 = data.loc[data['class'] == 0.5]
data2 = data.loc[data['class'] == 1]
print(len(data0), len(data1), len(data2))

data = pd.concat([data0, data1, data2], ignore_index=True)

'''
仅保留灯光
仅保留坡度
仅保留商场、医院、灯光
仅保留DEM、坡度、水体
排除高铁站、坡度、水体
排除商场、医院、灯光

[den_mall,dem,dist_roads,dist_water,dist_metrosta,dist_hwaysta,dist_railways,slope,dist_gov,dist_airpor,den_school,light_2010,den_spo,dist_railsta,den_hospital,class]
'''

driveFactor = data.columns.to_list()[:-1]
c = data.columns.to_list()[-1:]
data = data.sample(frac=1)
temp = driveFactor
temp.append('class')
['den_mall', 'dem', 'dist_roads', 'dist_water', 'dist_metrosta', 'dist_hwaysta', 'dist_railways', 'slope', 'dist_gov',
 'dist_airpor', 'den_school', 'light_2010', 'den_spo', 'dist_railsta', 'den_hospital', 'class']

"""
方案：逐步增加

'den_mall'


'dist_railsta'
程序运行时间:22.279635906219482毫秒
0.29921589994354625

'dist_railsta','dist_water'
程序运行时间:19.054937839508057毫秒
0.30413565733213344

'dist_railsta','dist_water','slope'
程序运行时间:19.1596999168396毫秒
0.37872897692274343

'dist_railsta','dist_water','slope','dist_gov'
程序运行时间:30.241466283798218毫秒
0.5584646280698805

'dist_railsta','dist_water','slope','dist_gov','den_spo'
程序运行时间:29.98125720024109毫秒
0.5593700348560728

'dist_railsta','dist_water','slope','dist_gov','den_spo','dist_hwaysta'
程序运行时间:31.24717378616333毫秒
0.6630209076629328

'dist_railsta','dist_water','slope','dist_gov','den_spo','dist_hwaysta','den_school'
程序运行时间:31.417532205581665毫秒
0.6766408131767698

'dist_railsta','dist_water','slope','dist_gov','den_spo','dist_hwaysta','den_school','light_2010','den_hospital'
程序运行时间:41.091846227645874毫秒
0.6777283507373156
"""
choose = [*[i for i in temp if i in ['dist_water', 'slope']], 'class']
# choose=[*[i for i in temp if i not in ['den_hospital','light_2010','den_mall']]]
comparisons = [
    [[*[i for i in temp if i in ['dist_water', 'slope']], 'class'], [*[i for i in temp if i in ['dem']], 'class']],
    [[*[i for i in temp if i in ['den_school', 'dist_roads', 'den_spo']], 'class'],[*[i for i in temp if i in ['dist_metrosta']], 'class']],
    [[*[i for i in temp if i in ['dem', 'dist_airpor']], 'class'],[*[i for i in temp if i in ['dist_metrosta']], 'class']],
    [[*[i for i in temp if i in ['', 'slope']], 'class'], [*[i for i in temp if i in ['dem']], 'class']],
    []

]

srcdata = data
for i in comparisons:
    choose = [i, 'class']
    data = srcdata[choose]
    print(i)
    train_data = data.sample(frac=0.7)
    test_data = data[~data.index.isin(train_data.index)]

    train_drive = train_data.drop('class', inplace=False, axis=1)
    train_tag = train_data['class'].astype('int')

    test_drive = test_data.drop('class', inplace=False, axis=1)
    test_tag = test_data['class'].astype('int')

    T1 = time.time()

    rf.fit(train_drive, train_tag)

    y_pred_rf = np.asarray(rf.predict(test_drive)).flatten()

    T2 = time.time()
    print('程序运行时间:%s毫秒' % (T2 - T1))

    import precision

    print(precision.cohen_kappa_score(y_pred_rf, test_tag))
