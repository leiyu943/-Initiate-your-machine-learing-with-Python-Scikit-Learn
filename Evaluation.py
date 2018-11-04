# -*- coding: utf-8 -*-
"""
    Created on Mon Jul 16 21:36:36 2018
    本示例给出用于机器学习结果评估的代码。一般原理参见matlab版代码的说明。
    scikit-learn提供了train_test_split函数来做随机划分，控制其参数可以实现按类别比例和不按比例的两种方法；    
    提供了KFold函数来做不按比例随机划分的交叉验证；
    提供了StratifiedKFold函数来做按类别比例划分的交叉验证
    @author: Xie Lingyun
    """

import numpy as np
from scipy.io import arff
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix

path = 'E:/Data/music_emotion.arff'
data, meta = arff.loadarff(path)
sample_num = len(data)
feature_num = len(data[0])-1 #因为arff文件数据部分的最后一维是标签，所以计算特征数目时要减去
tmpdata = []
for n in range(len(data)):
    tmpdata.append(list(data[n]))  #loadarff读出来的数据格式不是list，为了应用方便，先在此处转为list格式
mydata = []
mylabel = np.ones(sample_num, dtype=np.int16)
for i in range(sample_num):
    mydata.append(tmpdata[i][0:feature_num])
    mylabel[i] = tmpdata[i][feature_num]


lc = len(np.unique(mylabel))
train_cm = np.zeros([lc,lc],dtype=int)
test_cm = np.zeros([lc,lc],dtype=int)
test_acc = 0
train_acc = 0
clf = svm.SVC()

EvaluationMethod = 0  # 0-简单的随机划分测试集和训练集来评估分类器  1-交叉验证评估分类
if EvaluationMethod == 0:
    #方法1: 简单随机将数据分为训练集和测试集，测试集的比例可以通过test_size设定
    #train_test_split的stratify参数是按类别比例分配的开关，如果让它等于一个标签序列，就是按这个序列的
    #类别比例来拆分数据集的意思；如果不设置它，默认是任意随机拆分。此处给出了两种实现。
    train_data, test_data, train_label, test_label = train_test_split(mydata, mylabel, test_size=0.3, stratify=mylabel)
    #train_data, test_data, train_label, test_label = train_test_split(mydata, mylabel, test_size=0.3)
    
    clf.fit(train_data,train_label)
    train_pred = clf.predict(train_data)
    train_acc_sum = (train_pred == train_label).sum()
    train_acc = train_acc_sum/len(train_pred)
    train_cm = confusion_matrix(train_label, train_pred)
    test_pred = clf.predict(test_data)
    test_acc_sum = (test_pred == test_label).sum()
    test_acc = test_acc_sum/len(test_pred)
    test_cm = confusion_matrix(test_label, test_pred)

elif EvaluationMethod == 1:
    kfold = 4
    #方法2：KFolds交叉检验。这里也有按类别比例划分和不按比例划分的两种实现
    #kf = KFold(n_splits=kfold, shuffle=True)
    #for train, test in kf.split(mydata):
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)
    for train, test in skf.split(mydata,mylabel):
        x = mydata[train]
        y = mylabel[train]
        tx = mydata[test]
        ty = mylabel[test]
        clf.fit(x,y)
        pred = clf.predict(x)
        train_acc = train_acc + (pred == y).sum()/len(y)
        train_cm = train_cm + confusion_matrix(y, pred)
        pred = clf.predict(tx)
        test_acc = test_acc + (pred == ty).sum()/len(ty)
        test_cm = test_cm + confusion_matrix(ty, pred)
    train_acc = train_acc/kfold
    test_acc = test_acc/kfold

#结果输出
print("训练集的分类正确率: %f%%" %(100*train_acc))
print('训练集的混淆矩阵：')
print(train_cm)
print("\n测试集的分类正确率: %f%%" %(100*test_acc))
print('测试集的混淆矩阵：')
print(test_cm)

