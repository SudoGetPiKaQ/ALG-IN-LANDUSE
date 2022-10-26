from typing import List

import pandas as pd
from pgmpy.models import BayesianModel

data = pd.read_csv(r'../数据集/landUse_ent.csv', index_col=False)
data0 = data.loc[data['class'] == 0]
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

import plot.src as src

edge_tan = [('light_2010', 'den_school'), ('light_2010', 'den_spo'), ('light_2010', 'dist_metrosta'),
            ('light_2010', 'dist_roads'), ('light_2010', 'dist_railsta'), ('light_2010', 'dist_hwaysta'),
            ('light_2010', 'dem'), ('den_school', 'den_hospital'), ('dist_metrosta', 'dist_airpor'),
            ('dist_roads', 'dist_gov'), ('dist_railsta', 'dist_railways'), ('dem', 'dist_water'), ('dem', 'slope'),
            ('den_hospital', 'den_mall'), ('class', 'den_mall'), ('class', 'dem'), ('class', 'dist_roads'),
            ('class', 'dist_water'), ('class', 'dist_metrosta'), ('class', 'dist_hwaysta'), ('class', 'dist_railways'),
            ('class', 'slope'), ('class', 'dist_gov'), ('class', 'dist_airpor'), ('class', 'den_school'),
            ('class', 'light_2010'), ('class', 'den_spo'), ('class', 'dist_railsta'), ('class', 'den_hospital')]

model_TAN = BayesianModel(edge_tan)

model_TAN.fit(train_data)

from pgmpy.inference import VariableElimination, BeliefPropagation

model_infer = inference = BeliefPropagation(model_TAN)
li=src.toList
li.remove('class')
print(li)
q = model_infer.map_query(variables=li[:5], evidence={'class': 0},show_progress=True)
#{'den_mall': 1, 'dem': 4, 'dist_roads': 3, 'dist_water': 2, 'dist_metrosta': 29}

print(q)


# [print(i) for i in model_TAN.get_cpds()]



