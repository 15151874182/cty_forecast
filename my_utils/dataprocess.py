import os,sys
import datetime
import math
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib
from tqdm import tqdm
def points_to_days(df,cols=['tmp','target']):
    '''
    96个点统计成1天

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.
    cols : list
        需要统计max,min,mean的列名. The default is ['tmp','target'].

    Returns
    -------
    df_days : dataframe
        DESCRIPTION.

    '''
    df_days=[]
    from my_utils.tools import divide_by_n
    res=divide_by_n(df,96)
    for day in tqdm(res):
        for col in cols:
            day['max_'+col]=day[col].max()
            day['min_'+col]=day[col].min()
            day['mean_'+col]=day[col].mean()
        xx=day.iloc[0]
        xx['date']=xx['date'].split()[0]##去掉小时分钟
        for col in cols: 
            del xx[col]
        df_days.append(xx)
    df_days=pd.concat(df_days,axis=1).T
    df_days=df_days.reset_index(drop=True)
    return df_days