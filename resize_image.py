# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 18:14:12 2018

@author: 12088_000
"""

from __future__ import division  
from skimage import data,color,io,feature
from skimage.transform import resize

import numpy as np
import os
import scipy
import csv
#file='D:/美感分类实验信号-分组/气势美/2.85_l212.jpg'
path1='D:/美感分类实验信号-分组/'
lei='气势美','生机美','雅致美','萧瑟美','无法分类','清幽美'
zong=[]

for leibie in lei:
    path=path1+leibie+'/'
    filelist=os.listdir(path)
    for imname in filelist:
        file=path+imname
        image=io.imread(file)    
        imhsv=color.rgb2hsv(image)
        image_resized = resize(imhsv, (18,20))
        arr=image_resized.flatten()
        hang=list(arr)
        hang.insert(0,imname)
        hang.append(leibie)        
        zong.append(hang)

csvFile = open("D:/feature/im_resize90_100.csv", "w+")
writer = csv.writer(csvFile)
for item in zong:
    writer.writerow(item)
csvFile.close()
