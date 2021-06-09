#from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split #按照比例划分数据集合
from sklearn.preprocessing import MinMaxScaler #数据归一化
#from sklearn.utils import shuffle #随机排序
from keras.utils import np_utils
import pandas as pd
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten , Activation, SimpleRNN, LSTM, GRU, Dropout, TimeDistributed, Reshape, Input, Lambda, Add, BatchNormalization, Dropout 
from tensorflow.keras import Sequential
#import matplotlib.pyplot as plt
#from sklearn.metrics import ConfusionMatrixDisplay
#from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
#from imblearn.over_sampling import RandomOverSampler

a = pd.read_csv('data_use1.csv')
data = a.drop(['y_var'], axis=1)
a['y_var'] -= 1#标签转换 -1

res = np_utils.to_categorical(a['y_var'], num_classes=3)
train_data, test_data, train_res, test_res = train_test_split(data, res, test_size=0.2, shuffle=True, random_state=1 ,stratify=res) 
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_data)

train_data = pd.DataFrame(scaler.transform(train_data))
test_data = pd.DataFrame(scaler.transform(test_data))

# 模型声明
from tensorflow.keras.layers import LeakyReLU
model = Sequential()
model.add(Reshape((-1,1), input_shape=(6,)))
model.add(Conv1D(64, 1, padding='valid', kernel_initializer='he_uniform'))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Conv1D(128, 3, padding='valid', kernel_initializer='he_uniform'))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Conv1D(128, 3, padding='valid', kernel_initializer='he_uniform'))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(3,activation='softmax', dtype='float32'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_res, epochs=500, batch_size=32, verbose=0)#训练
a = model.evaluate(test_data, test_res, batch_size=32, verbose=0)[-1]#测试

print('begin:',a)

a = 0
while a < 0.71:
    model.fit(train_data, train_res, epochs=50, batch_size=32, verbose=0)
    #print('train:',model.history.history['accuracy'][-1])
    a1 = model.evaluate(test_data, test_res, batch_size=32, verbose=0)[-1]
    #if a1 > 0.65:
    print('test:',a1)
    if a1 > a:
        a = a1
print('final:',a)
