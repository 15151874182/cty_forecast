'''
2种归一化
'''
import pandas as pd
from scipy.signal import argrelextrema
from scipy import interpolate
import numpy as np

def IMF(data,col,length = 96):
    """
    功能：
        寻找输入时间序列的IMF(一种平滑方法)
    输入：
        data 输入的负荷时间序列,['datetime',col]
        length = 96  设置区间长度,用于寻找区间极值
    返回：
        包含IMF的一张总表，Dataframe
    """
    df = data.copy(deep=True)
    df.loc[:,'t'] = df.index
    
    #找到每个区间内中的极大值、极小值及对应的时刻t
    max_index = []
    min_index = []
    
    #找到区间极值的索引值
    for day in range(len(df)//length):
        temp = df.iloc[day*length:(day+1)*length,:]
        max_index.append(temp[col].idxmax(axis=0))
        min_index.append(temp[col].idxmin(axis=0))
    
    max_value = df.loc[max_index,[col,"t"]]
    min_value = df.loc[min_index,[col,"t"]]
    #处理极大值插值
    #定义一个三次样条插值方法的函数
    cb = interpolate.interp1d(max_value['t'].values,\
                              max_value[col].values,kind='cubic')
    #定义插值区间t
    t_range = np.arange(max_value['t'].min(), max_value['t'].max(), 1)
    
    #区间范围外的插值我们设置为实际值,并将插值的值写入到总表中
    df.loc[:,'interp_max'] = df[col] 
    df.loc[t_range,'interp_max'] = cb(t_range)
    
    
    #处理极小值插值
    #定义一个三次样条插值方法的函数
    cb = interpolate.interp1d(min_value['t'].values,\
                              min_value[col].values,kind='cubic')
    #定义插值区间t
    t_range = np.arange(min_value['t'].min(), min_value['t'].max(), 1)
    #区间范围外的插值我们设置为实际值,并将插值的值写入到总表中
    df.loc[:,'interp_min'] = df[col] 
    df.loc[t_range,'interp_min'] = cb(t_range)
   
    #求出上下包络线的平均值m(t)，在原时间序列中减去它:h(t)=x(t)-m(t)
    df.loc[:,'mean'] = (df['interp_max'].values +df['interp_min'].values )/2
    df.loc[:,'h(t)'] = (df[col].values -df['mean'].values )
    
    return df