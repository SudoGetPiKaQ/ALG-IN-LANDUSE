import pandas as pd
from pgmpy.models import BayesianModel
import networkx as nx
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch, BicScore


data = pd.read_csv('../数据集/landUse_continue.csv',index_col=False)
driveFactor = data.columns.to_list()[:-1]
c = data.columns.to_list()[-1:]
data=data.sample(frac=0.005)
model = BayesianModel([(c[0], i) for i in driveFactor])



print(data)