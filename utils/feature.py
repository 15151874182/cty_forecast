'''

'''
import pandas as pd
import numpy as np

def date_to_timeFeatures(df):
    ###经测试，HourOfDay和DayOfWeek能提升效果，其它反而会干扰
    # df['SecondOfMinute']=df['date'].apply(lambda x : x.second / 59.0 - 0.5)
    # df['MinuteOfHour']=df['date'].apply(lambda x : x.minute / 59.0 - 0.5)
    df['HourOfDay']=df['date'].apply(lambda x : x.hour / 23.0 - 0.5)
    df['DayOfWeek']=df['date'].apply(lambda x : x.dayofweek / 6.0 - 0.5)
    # df['DayOfMonth']=df['date'].apply(lambda x : (x.day - 1) / 30.0 - 0.5)
    # df['DayOfYear']=df['date'].apply(lambda x : (x.dayofyear - 1) / 365.0 - 0.5)
   
    return df

def create_new_feas(df):
    today_past_96_list=[]
    for row in range(96,len(df)):
        today_past_96=df['today'].iloc[row-96:row]
        today_past_96_list.append(today_past_96)
    # today_past_96_list=list(reversed(today_past_96_list))##day-1point在最前面，所以要颠倒顺序
    today_past=np.asarray(today_past_96_list)
    today_past=pd.DataFrame(today_past).reset_index(drop=True)
    cols_name=['day-{}point'.format(i) for i in range(96,0,-1)]
    today_past.columns=cols_name
    df2=df.iloc[96:].reset_index(drop=True)
    df=pd.concat([df2,today_past],axis=1)    
    
    # df['day-1/day-2']=df['day-1']/df['day-2']#12个月总平均0.957,略高于base,day-1/day-2特征可以用,对于节假日也有一点点作用
    # df['day-2/day-3']=df['day-2']/df['day-3']#12个月总平均0.9565,略微下降,day-2/day-3整体虽然下降，但是对20年2月节假日有些提升
    # df['day-3/day-4']=df['day-3']/df['day-4']#12个月总平均0.9567,略微下降,对20年2月节假日也无帮助，可删除
    # df['tmp/today_tmp']=df['tmp']/df['today_tmp']#12个月总平均0.9563，略微下降,'tmp/today_tmp'特征分母可能有0，造成干扰，可删除,下一个尝试tmp-today_tmp
    # df['tmp-today_tmp']=df['tmp']-df['today_tmp']#12个月总平均0.9568,略微下降，21年9月变好，20年8月变差，有些月份可能有用

    return df

def fea_shift(df,name='load',N=8):
    for i in range(1,N+1):
        # print(i)
        df['{}-{}-point'.format(name,i)]=df[name].shift(i)
    return df

def wd_to_sincos_wd(df,wd_cols,delete=True):
    for col in wd_cols:        
        df[f'{col}_sin']=np.sin(np.pi*df[f'{col}']/180)
        df[f'{col}_cos']=np.cos(np.pi*df[f'{col}']/180)
        if delete==True:
            del df[f'{col}']
    return df