import numpy as np
import sys
from numpy.core.fromnumeric import shape
import scipy.io as scio
import random


import mainFunctionMistake
import mainFunctionExampleLabelBased

from set_random_seed import set_seed
import math

#导入数据集
datasetname='medical'
dataset=scio.loadmat(datasetname)

data=dataset['data']
target=dataset['target']


#数据集参数
num_ins=np.size(data,0)
num_fea=np.size(data,1)
num_class=np.size(target,0)


#对uint16型的data进行类型转换
temp_data=np.zeros(shape=[num_ins,num_fea])
for i in range(0,num_ins):
    for j in range(0,num_fea):
        temp_data[i,j]=float(data[i,j])
data=temp_data


#特征标准化
for n in range(0,num_fea):
    temp=data[:,n]
    max_temp=max(temp)
    min_temp=min(temp)
    if max_temp!=min_temp:
        for i in range(0,num_ins):
            data[i][n]=(data[i][n]-min_temp)/(max_temp-min_temp)

# mistake_mode：标注错误的种类。0 代表没有错误。
# 0 no mistake / 1 difficult positive / 2 easy positive / 3 difficult negative / 4 easy negative
for mistake_mode in range(0,1):  
    if mistake_mode==0:
        num_mistake_list=[0] # 每个instance最多标注错误的label的个数
    else:
        num_mistake_list=[1,3,5]
    for num_mistake in num_mistake_list:
        #k = nfold折交叉验证
        nfold=1
        for seed in range(1,nfold+1):
            print('Cross Validation Round For:',seed)
            set_seed(seed)

            #划分训练集和测试集 50%训练集和50%测试集
            train_test_rate=0.5
            train_test_index=list(range(0,num_ins))
            random.shuffle(train_test_index)

            train_data_index=train_test_index[0:round(train_test_rate*num_ins)]
            test_data_index=train_test_index[round(train_test_rate*num_ins):num_ins]

            train_data=data[train_data_index,:]
            train_target=target[:,train_data_index]

            test_data=data[test_data_index,:]
            test_target=target[:,test_data_index]

            #在训练集中划分标注集和未标注集 已标注集5% 未标注集95%
            labelled_rate=0.05
            num_train=np.size(train_data,0)
            labelled_unlabelled_index=list(range(0,num_train))
            random.shuffle(labelled_unlabelled_index)
            num_labelled=round(labelled_rate*num_train)

            labelled_index=labelled_unlabelled_index[0:num_labelled]
            unlabelled_index=labelled_unlabelled_index[num_labelled:num_train]

            labelled_data=train_data[labelled_index,:]
            labelled_target=train_target[:,labelled_index]

            unlabelled_data=train_data[unlabelled_index,:]
            unlabelled_target=train_target[:,unlabelled_index]


            lr_rate = 8 # learning rate
            wd_rate = 1 # weight decay
            query_mode=6 #1 random query / 2 MMU query / 3 LCI / 4 CVIRS /5 RMLAL /6 AUDI(instance-label pair based)
            

            difficult_threshold=1/4 #the threshold between difficult labels and easy labels
            
            #记录实验结果
            num_unlabelled=unlabelled_data.shape[0]
            if seed==1:
                HanRecord=np.zeros([nfold,math.floor(num_unlabelled/2)+1])
                RanRecord=np.zeros([nfold,math.floor(num_unlabelled/2)+1])
                CovRecord=np.zeros([nfold,math.floor(num_unlabelled/2)+1])
                AvgRecord=np.zeros([nfold,math.floor(num_unlabelled/2)+1])
                MicRecord=np.zeros([nfold,math.floor(num_unlabelled/2)+1])
                AnnRecord=np.zeros([nfold,math.floor(num_unlabelled/2)+1])
                MisRecord=np.zeros([nfold,num_unlabelled+1])
                DifRecord=np.zeros([nfold,num_class])


            # instance based 和 instance-label pair based 分开
            if query_mode<6:
                [Model,HammingLoss,RankingLoss,Coverage,Average_Precision,MicroF1,AnnotationCost,Mistake,TargetDiffer] = mainFunctionMistake.ActiveLearning(labelled_data,labelled_target,unlabelled_data,unlabelled_target,test_data,test_target,lr_rate,wd_rate,query_mode,mistake_mode,num_mistake,difficult_threshold,seed)
            else:
                [Model,HammingLoss,RankingLoss,Coverage,Average_Precision,MicroF1,AnnotationCost,Mistake,TargetDiffer] = mainFunctionExampleLabelBased.ActiveLearning(labelled_data,labelled_target,unlabelled_data,unlabelled_target,test_data,test_target,lr_rate,wd_rate,num_mistake,difficult_threshold,seed)

            HanRecord[seed-1,:]=HammingLoss
            RanRecord[seed-1,:]=RankingLoss
            CovRecord[seed-1,:]=Coverage
            AvgRecord[seed-1,:]=Average_Precision
            MicRecord[seed-1,:]=MicroF1
            AnnRecord[seed-1,:]=AnnotationCost
            MisRecord[seed-1,:]=Mistake
            DifRecord[seed-1,:]=TargetDiffer

            #保存结果
            filename='num_mistake='+str(num_mistake)+'_process='+str(seed)+'.mat'
            scio.savemat(filename,{
                'seed':seed
            })

        #对十次求均值
        HanAverage=np.mean(HanRecord,axis=0)
        RanAverage=np.mean(RanRecord,axis=0)
        CovAverage=np.mean(CovRecord,axis=0)
        AvgAverage=np.mean(AvgRecord,axis=0)
        MicAverage=np.mean(MicRecord,axis=0)
        AnnAverage=np.mean(AnnRecord,axis=0)
        MisAverage=np.mean(MisRecord,axis=0)
        DifAverage=np.mean(DifRecord,axis=0)

        #对十次求标准差
        HanStd=np.std(HanRecord,axis=0)
        RanStd=np.std(RanRecord,axis=0)
        CovStd=np.std(CovRecord,axis=0)
        AvgStd=np.std(AvgRecord,axis=0)
        MicStd=np.std(MicRecord,axis=0)
        AnnStd=np.std(AnnRecord,axis=0)
        MisStd=np.std(MisRecord,axis=0)
        DifStd=np.std(DifRecord,axis=0)



        Filename='Mistake_'+datasetname+'_query='+str(query_mode)+'_mistake='+str(mistake_mode)+'_num_mistake='+str(num_mistake)+'_difficult_threshold'+str(difficult_threshold)+'.mat'
        scio.savemat(Filename,{
            'HanRecord':HanRecord,
            'RanRecord':RanRecord,
            'CovRecord':CovRecord,
            'AvgRecord':AvgRecord,
            'MicRecord':MicRecord,
            'AnnRecord':AnnRecord,
            'MisRecord':MisRecord,
            'DifRecord':DifRecord,
            'HanAverage':HanAverage,
            'RanAverage':RanAverage,
            'CovAverage':CovAverage,
            'AvgAverage':AvgAverage,
            'MicAverage':MicAverage,
            'AnnAverage':AnnAverage,
            'MisAverage':MisAverage,
            'DifAverage':DifAverage,
            'HanStd':HanStd,
            'RanStd':RanStd,
            'CovStd':CovStd,
            'AvgStd':AvgStd,
            'MicStd':MicStd,
            'AnnStd':AnnStd,
            'MisStd':MisStd,
            'DifStd':DifStd
        })




print('end')