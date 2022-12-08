lgb
wind1数据集

目标：从8个变量中找到test上最佳特征组合

all feas:  ['date', 'ws30', 'wd30', 'ws50', 'wd50', 'ws70', 'wd70', 't_50', 'p_50', 'target']

先用lgb做单变量实际测试：

['ws30', 'wd30', 'ws50', 'wd50', 'ws70', 'wd70', 't_50', 'p_50']——用全部feas——loss: 1546

[''ws70'']——loss: 1471   [''ws50'']——loss: 1569   [''ws30'']——loss: 1757

[''p_50']——loss: 1732    [''t_50'']——loss: 2009     [''wd50'']——loss: 2099

[''wd70'']——loss: 2089   [''wd30'']——loss: 2085  

pearson

![image-20221205150021037](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221205150021037.png)

pearson和单变量结果基本正相关，且['ws70']单变量反而超过使用所有变量，说明冗余变量对lgb的坏处



组合变量测试:

['ws30', 'ws50', 'ws70', 't_50', 'p_50']——删除了无用的wdXX变量——loss: 1433

['ws30', 'ws50', 'ws70', 'p_50']——删除了弱相关的t_50变量——loss: 1285

['ws30', 'ws70', 'p_50']——删除了强相关但是冗余的ws50变量——loss: 1272