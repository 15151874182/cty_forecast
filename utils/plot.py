import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import datetime
import numpy as np
import copy

def plot_peroid(df,filename,time_col = "datetime",cols = ["actual","predict_load"],start_day = "2021-11-12",end_day=None,days = 30):
    """
    按指定日期进行画图
    start_day: 设置起始的日期
    days:从起始日开始，一共要画多少天
    """
    #将cols列全部转换成数值类型
    for col in cols:
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')
    
    #选择我们想要的列
    add_time = copy.deepcopy(cols)
    add_time.append(time_col)
    df = df.loc[:,add_time]
    
    #时间列转化成时间格式
    df[time_col] = pd.to_datetime(df[time_col])
    
    start_day = pd.to_datetime(start_day)
    
    #画图进行对比,设置画布大小
    plt.figure(figsize=(30,10))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False
    
    #取需要画图的时间段数据
    if end_day ==None:
        end_day = start_day + pd.Timedelta(days=days)
    else:
        end_day = pd.to_datetime(end_day)
    temp = df[np.logical_and((df[time_col]>=start_day),(df[time_col]<end_day))]
    #画图
    x = temp.loc[:,time_col]
    #画出所有的列
    for col in cols:
        plt.plot(x, temp.loc[:,col],label=col,alpha=0.6)
    print(f"画图：{filename}")
    plt.legend(loc="upper left",fontsize='x-large')
    plt.title(f"{filename}",fontsize='x-large')
    # plt.savefig(f"./figure/{filename}.png",dpi=300,bbox_inches='tight',pad_inches=0.0)
    plt.show()
    # plt.close()

def plot_without_date(df,filename,cols = ["actual","predict_load"]):

    #将cols列全部转换成数值类型
    for col in cols:
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')
    
    #选择我们想要的列
    df = df.loc[:,cols]

    #画图进行对比,设置画布大小
    plt.figure(figsize=(30,10))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False
    
    #画出所有的列
    for col in cols:
        plt.plot(df.loc[:,col],label=col,alpha=0.6)
    print(f"画图：{filename}")
    plt.legend(loc="upper left",fontsize='x-large')
    plt.title(f"{filename}",fontsize='x-large')
    # plt.savefig(f"./figure/{filename}.png",dpi=300,bbox_inches='tight',pad_inches=0.0)
    plt.show()
    # plt.close()