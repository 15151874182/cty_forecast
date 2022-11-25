# -*- coding: utf-8 -*-
import os,sys
import datetime
import math
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib


if __name__=='__main__':
    ##项目目录加入环境变量
    project_path=os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_path)
    ####Step1:read raw data
    from dataset import Dataset
    dataset=Dataset()
    df=dataset.load_busbar1_cluster().data
    df=df.dropna()
    
    df1=df.iloc[:,:200]
    df1['bus_id']=df['bus_id'] ##270*(200+bus_id)
    df1['bus_id']=df1['bus_id'].apply(lambda x: str(x))
    
    bus_id=df.pop('bus_id')
    statistic=df.T.describe().T  ##计算出统计量
    del statistic['count']   ##数量没用
    from sklearn.preprocessing import StandardScaler
    stand_scaler = StandardScaler()
    xx = stand_scaler.fit_transform(statistic) ##中心化
    from sklearn.decomposition import PCA  
    p_scaler = PCA(n_components='mle') #手动确定
    xx = p_scaler.fit_transform(xx)  ##降维
    xx = stand_scaler.fit_transform(xx) ##xx是降维后，中心化后的中间产物
    
    ###下面代码可以看图挑选出合适的n_clusters
    # losses = []  # 存放每次结果的误差平方和
    # for k in range(1,10):
    #     print(k)
    #     k_scaler=kmeans(df,n_clusters=k)
    #     losses.append(k_scaler.inertia_)
    # X = range(1,10)
    # plt.xlabel('k')
    # plt.ylabel('SSE')
    # plt.plot(X,losses,'o-')
    
    from sklearn.cluster import KMeans
    k_scaler=KMeans(n_clusters=5,random_state=1)
    k_scaler.fit(xx)
    ##聚类效果可视化
    # colors = ['r', 'y', 'g', 'b', 'c', 'k'] ##长度要大于label数量
    # for id, label in enumerate(k_scaler.labels_):
    #     plt.scatter(xx[id][0], xx[id][1], color = colors[label],marker='o',s=4)
    # plt.show()
    
    ####聚类结果res
    bus_id=pd.DataFrame(bus_id)
    bus_id=bus_id.reset_index(drop=True)
    label=pd.DataFrame(k_scaler.labels_)
    res=pd.concat([bus_id,label],axis=1) ##给label赋予bus_id
    res['bus_id']=res['bus_id'].apply(lambda x: str(x))
    res.columns=['bus_id','kmeans_label']
    
    ##wulin对比结果
    duibi=dataset.load_busbar1_duibi().data ##wulin对比csv
    duibi=duibi[['area_id','acc_span']]
    duibi.columns=['bus_id','acc']
    duibi=duibi.dropna(how='any')
    duibi['bus_id']=duibi['bus_id'].apply(lambda x: x.replace('\'',''))
    
    ##merge生成最终的output
    output=res.merge(duibi,on='bus_id').merge(df1,on='bus_id')
    output['bus_id']=output['bus_id'].apply(lambda x: x+'\t')
    output.to_csv('label_acc.csv')
    plt.scatter(output['kmeans_label'],output['acc'])
    plt.show()
    
    ##聚类选择预测不好的母线可视化
    # bad_df=output[output['kmeans_label']==1]
    # bad_df=bad_df.iloc[:,3:]
    # for i in range(0,len(bad_df),8):
    #     plt.plot(bad_df.iloc[i])
    #     plt.show()
    

