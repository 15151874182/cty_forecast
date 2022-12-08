# -*- coding: utf-8 -*-
import os,sys
import datetime
import math
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib

project_path=os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_path)
from dataset import Dataset
dataset=Dataset()
# df=dataset.load_system_tang(mode='ts').data
# df=dataset.load_system1(mode='ts').data
df=dataset.load_wind1().data
del df['date']
####1.1 方差选择法，返回值为特征选择后的数据#参数threshold为方差的阈值
# from sklearn.feature_selection import VarianceThreshold
# selector=VarianceThreshold(threshold=3)
# res=selector.fit_transform(df)
def pearson_coe(df,label_name='load'):####label_name是目标的名字，比如'load','nextday'
    # del df['datetime']
    label=df[label_name]
    pearson_dict={}
    for f in df.columns:
        k=label.corr(df[f],method="pearson") #皮尔森相关性系数
        pearson_dict['{}-{}'.format(label_name,f)]=k
    pearson_dict=sorted(pearson_dict.items(),key=lambda x:abs(x[1]),reverse=True)
    print(pearson_dict)

# pearson_coe(df,label_name='target')
from sklearn import metrics
for i in ['ws70','ws50','ws30','p_50','t_50','wd50']:
    print(metrics.mutual_info_score(df['target'],df[i]))