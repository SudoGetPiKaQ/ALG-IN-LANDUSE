import numpy as np
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt
import os
import precision

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 读取训练数据和测试数据
data = pd.read_csv('../数据集/landUse_ent.csv', index_col=False)
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

model = keras.Sequential()
# Flatten层会把除了第一维(样本数)之外的,也就是后面的维度摊平成同一个维度里
# model.add(layers.Flatten()) # (60000, 28, 28) => (60000, 28*28)
# 已经不是第一层了,所以不再需要input_dim了,会直接根据前面层的输出来设置
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dropout(0.5))  # 添加Dropout
model.add(layers.Dense(3, activation='softmax'))
model.compile(
    optimizer='adam',
    # 注意因为label是顺序编码的,这里用这个
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

history = model.fit(np.asarray(train_data.drop('class', axis=1, inplace=False)), np.asarray(train_data['class']), epochs=50, batch_size=1024,
                    validation_data=(np.asarray(predict_data), np.asarray(verifyData)), verbose=0)
y_pred=model.predict_classes(np.asarray(predict_data))

# model.evaluate(np.asarray(train_data.drop('class', axis=1, inplace=False)), np.asarray(train_data['class']))
#
# plt.plot(history.epoch, history.history.get('val_acc'), c='g', label='validation acc')
# plt.plot(history.epoch, history.history.get('acc'), c='b', label='train acc')
# plt.legend()
# plt.show()
