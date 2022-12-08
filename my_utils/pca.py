# -*- coding: utf-8 -*-
import os,sys
import datetime
import math
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

if __name__=='__main__':
    ##项目目录加入环境变量
    project_path=os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_path)
    ####Step1:read raw data
    from dataset import Dataset
    dataset=Dataset()
    # df=dataset.load_system_tang(mode='ts',y_shift=96).data
    df=dataset.load_busbar1_cluster().data
    df=df.dropna()
    del df['bus_id']
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    df = minmax_scaler.fit_transform(df.T) ##minmax_scaler只能列归一化，行归一化得转置
    df=df.T ##转置回来
    # df.interpolate(method="linear",axis=1,inplace=True)
    
    ##p_scaler = PCA(n_components='mle') 极大似然估计自己确定n_components，但要n_samples >= n_features   
    # p_scaler = PCA(n_components=0.95, svd_solver='full') ##保留95%的信息
    p_scaler = PCA(n_components=2) #手动确定
    res = p_scaler.fit_transform(df)
    print(res.shape)