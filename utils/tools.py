'''
2种归一化
'''
import pandas as pd
import numpy as np
import random
random.seed(10)

def divide_by_n(df,n):
    ##df要是n的倍数，如果不满足，可以先处理一下
    days=int(df.shape[0]/n)##102624个点数据，1天96个点*1069天
    df_n_list=[]##存储每个df_n块，长度=天数
    for i in range(days):
        df_n=df.iloc[i*n:(i+1)*n,:]##整体df按天划分出1069个df_96
        df_n_list.append(df_n)
    return df_n_list

####将数据集按比例划分为train,val,test
def dataset_split(df,n,ratio=[0.7,0.2,0.1]):
    '''

    Parameters
    ----------
    df : dataframe
        源数据集.
    n : 每天包含几个样本
        DESCRIPTION.
    ratio : list, optional
        train,val,test划分比例. The default is [0.7,0.2,0.1].

    Returns
    -------
    trainset : dataframe
        DESCRIPTION.
    valset : dataframe
        DESCRIPTION.
    testset : dataframe
        DESCRIPTION.

    '''
    ##划分逻辑：先按天划分，计算出train，val，test分别占哪几天
    ##然后根据天数计算出每段id在df种的具体ids，并提取对应数据
    blocks=int(len(df)/n) ##按天分成几块
    train_len=int(blocks*ratio[0]) ##train占几个块
    val_len=int(blocks*ratio[1])
    
    index=[i for i in range(blocks)]
    train_id=random.sample(index,train_len)  
    index=list(set(index)-set(train_id))
    val_id=random.sample(index,val_len)
    test_id=list(set(index)-set(val_id))

    train_ids=[]
    val_ids=[]
    test_ids=[]
    for id in train_id:
        train_ids+=[i for i in range(n*id,n*(id+1))]
    for id in val_id:
        val_ids+=[i for i in range(n*id,n*(id+1))]
    for id in test_id:
        test_ids+=[i for i in range(n*id,n*(id+1))]
        
    trainset=df.iloc[train_ids]
    valset=df.iloc[val_ids]
    testset=df.iloc[test_ids]
      
    return trainset,valset,testset