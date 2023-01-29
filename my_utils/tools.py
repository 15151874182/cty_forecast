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
def dataset_split(df,n,ratio=[0.7,0.2,0.1],mode=1):
    '''
    支持时序数据的按组打乱，保持组内数据的时序性
    Parameters
    ----------
    df : dataframe
        源数据集.
    n : 每组包含几个样本(一个bin里面有几个样本),支持n=1,和普通数据划分一致
        n可以不被len(df)整除，但会丢失一些数据
        DESCRIPTION.
    ratio : list
        train,val,test划分比例. The default is [0.7,0.2,0.1].
    mode : int   1.test抽取最后部分不打乱，train,val等间隔采样，train保持一定组外时序性
                 2.test抽取最后部分不打乱，train,val 打乱组外时序性
                 划分原则：test方便画图看图，不需要打乱；也不需要一定保持和train同分布
                         train和val必须同分布，这样val的中间结果才有泛化性，能辅助模型调整
        数据划分方式.
    Returns
    -------
    trainset : dataframe
        DESCRIPTION.
    valset : dataframe
        DESCRIPTION.
    testset : dataframe
        DESCRIPTION.

    '''
    ##划分逻辑：先按组划分，计算出train，val，test分别占哪几组
    ##然后根据天数计算出每段id在df种的具体ids，并提取对应数据
    bins=int(len(df)/n) ##先按组划分
    test_len=int(bins*ratio[2]) ##test占几组
    val_len=int(bins*ratio[1])   ##val占几组
    # train_len=bins-test_len-val_len ##train占几组
    
    index=[i for i in range(bins)]
    test_id=index[-test_len:] ##test先抽最后面
    index=index[:-test_len]   ##index去掉test
    gap=len(index)//val_len   ##根据抽样数，计算间隔gap
    val_id=[gap//2+gap*i for i in range(val_len)] ##等间隔抽样
    train_id=list(set(index)-set(val_id))##index从中拿掉等间隔的val，生成train
    
    if mode==2: ##上面结果已经是mode=1了，mode=2打乱即可
        random.shuffle(train_id)
        random.shuffle(val_id)

    
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

####皮尔森相关系数
def pearson_coe(df,label_name='load'):####label_name是目标的名字，比如'load','nextday'
    # del df['datetime']
    label=df[label_name]
    pearson_dict={}
    for f in df.columns:
        k=label.corr(df[f],method="pearson") #皮尔森相关性系数
        pearson_dict['{}-{}'.format(label_name,f)]=k
    pearson_dict=sorted(pearson_dict.items(),key=lambda x:abs(x[1]),reverse=True)
    print(pearson_dict)