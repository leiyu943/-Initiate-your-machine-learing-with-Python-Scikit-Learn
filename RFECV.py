# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:13:47 2018

@author: Xie Lingyun
"""

import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from scipy.io import arff

#读取arff文件的数据
def readarff(path):
    data, meta = arff.loadarff(path)
    sample_num = len(data)
    feature_num = len(data[0])-1
    tmpdata = []
    for n in range(len(data)):
        tmpdata.append(list(data[n]))
        x_data = []
        x_label = np.ones(sample_num, dtype=np.int16)
    for i in range(sample_num):
        x_data.append(tmpdata[i][0:feature_num])
        x_label[i] = tmpdata[i][feature_num]
    return x_data, x_label

train_data, train_label = readarff('D:/feature/feature0811 - 158 features.arff')
#test_data, test_label = readarff('E:/PythonCode/mxy/test.arff')
#train_data.extend(test_data)
#train_label = np.append(train_label,test_label)
'''
#此处是调用pandas的读取csv格式的函数，中间设定了数据格式，同时把
#标签那一列读为int类型，设定标签列为第90列
path = 'E:/Data/Drawing/feature0720-5lei.csv'
data = pd.read_csv(path,dtype = np.float32, converters={'class':int},index_col=90)
train_data = data.values
train_label = np.array(data.index)
'''
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,5))
train_data = min_max_scaler.fit_transform(train_data)
#数据范围的尺度变换
#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,8))
#train_data = min_max_scaler.fit_transform(train_data)
#test_data = min_max_scaler.fit_transform(test_data)
print("data readed")

#特征选择并加以交叉验证，每次淘汰排名最后一个的特征
clf = svm.SVC(kernel='linear')
print("SVC loaded")
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5),
              scoring='accuracy')
print("RFECV finished")
rfecv.fit(train_data, train_label)

print("最佳特征数目为 : %d" % rfecv.n_features_)
x_label = range(1, len(rfecv.grid_scores_) + 1)
y_label = rfecv.grid_scores_
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross-Validation Score (Classification Accuracy)")
plt.plot(x_label, y_label)
plt.show()
