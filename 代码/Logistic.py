from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
import pandas as pd
import precision
# read data
data = pd.read_csv(r'../数据集/landUse_ent.csv', index_col=False)

data0 = data.loc[data['class'] == 0].sample(frac=0.05)
data1 = data.loc[data['class'] == 1]
data2 = data.loc[data['class'] == 2]
print(len(data0), len(data1), len(data2))

data = pd.concat([data0, data1, data2], ignore_index=True)

driveFactor = data.columns.to_list()[:-1]
c = data.columns.to_list()[-1:]
# data = data.sample(frac=1)

train_data = data.sample(frac=0.7)
test_data = data[~data.index.isin(train_data.index)]
verifyData = test_data['class']
predict_data = test_data.copy()
predict_data.drop('class', axis=1, inplace=True)
#定义模型
log_model = LogisticRegression(max_iter=1000)
#训练模型
log_model.fit(train_data.drop('class', axis=1, inplace=False),train_data['class'])

#预测数据
y_pred = log_model.predict(predict_data)

print(precision.cohen_kappa_score(y_pred, verifyData))