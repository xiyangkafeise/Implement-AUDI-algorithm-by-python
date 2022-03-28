# Implement-AUDI-algorithm-by-python
AUDI is an instance-label pair multi-label active learning method.

文章："Active Query Driven by Uncertainty and Diversity for Incremental Multi-Label Learning" 2013 IEEE ICDM

方法：（1）选择预测正标记数目与已标记样本cardinality差距最大的instance（对LCI算法进行拓展）；
      （2）选择（1）中instance未标注label中uncertainty最大的label进行标注

运行环境anaconda：
python 3.9.7
numpy 1.20.3
scipy 1.7.1
scikit-learn 0.24.2
pytorch 1.11.0

多标记神经网络：
简单的三层设计，一层隐藏层，全连接
损失函数采用二类交叉熵 Binary Crossentropy Loss，只计算已标注instance-label pair的loss

实验设置：
初始时划分50%训练集和50%测试集，训练集由5%的完全已标注样本集和95%的待标注样本集组成
每query并标注了2q个label时，训练一次神经网络，q是标记空间大小。
每次训练的停止条件即，每个epoch后在测试集上计算microF1，多次效果不提升则停止，这使得每次训练都让测试集上效果最佳。
训练神经网络可以在上次基础上继续训练，或者重新训练。

出现的问题：
每次继续训练神经网络时，非常容易过拟合
过拟合使得计算loss时，会出现预测的概率为0，真实标记为1的情况，或者反之。使得交叉熵为INF，无法继续。
但是过拟合时效果也很好，增加一些约束之后（增加weight decay），会让效果不好。