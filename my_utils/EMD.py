# -*- coding: utf-8 -*-

import os,sys
import datetime
import math
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib

from dataset import Dataset
dataset=Dataset()
df=dataset.load_system1().data
# df=dataset.load_wind1().data
# df=dataset.load_system1().data
df=df.dropna()
ts=df['target'][:100]

ts=np.array(ts)
# ts = 2*np.sin(2*np.pi*15*t) +4*np.sin(2*np.pi*10*t)*np.sin(2*np.pi*t*0.1)+np.sin(2*np.pi*5*t)
from PyEMD import EMD, Visualisation
emd = EMD()
emd.emd(ts)
imfs, res = emd.get_imfs_and_residue()
# xx=imfs[1]+imfs[2]+res ##将imfs[0]高频噪音去掉，得到平滑的曲线
# plt.plot(xx)
# 绘制 IMF
vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, include_residue=True)
# # 绘制并显示所有提供的IMF的瞬时频率
# vis.plot_instant_freq(t=t,imfs=imfs)
# vis.show()