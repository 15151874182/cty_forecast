数据集和划分方式，添加了一个特征工程方法，否则lstm每个周期开始几天效果不好

![image-20221220101037542](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220101037542.png)

1.LSTM：

![image-20221220101853723](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220101853723.png)

![(C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220101357332.png)

![image-20221220101244807](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220101244807.png)

![image-20221220102614635](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220102614635.png)

2.BiLSTM:

![image-20221220102006652](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220102006652.png)

![image-20221220102537566](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220102537566.png)

![image-20221220102737699](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220102737699.png)

效果比LSTM还差一点

3.Seq2seq:

![image-20221220103130286](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220103130286.png)

![image-20221220103201787](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220103201787.png)

![image-20221220104004929](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220104004929.png)

![image-20221220112300369](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220112300369.png)

相对LSTM有个0.2%的提升

4.Lightgbm:

optuna自动化调参

![image-20221220123131239](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220123131239.png)

![image-20221220123225783](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221220123225783.png)

和lstm差不多，但是速度快几个量级，对系统环境要求也低，适合实战部署

5.Attention:

![image-20221228161512642](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221228161512642.png)

![image-20221228161614353](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221228161614353.png)

![image-20221228161636852](C:\Users\cty\AppData\Roaming\Typora\typora-user-images\image-20221228161636852.png)

效果很差训练还巨慢，网上查了一些资料，attention并不适合做时序预测问题