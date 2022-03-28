import numpy as np
from numpy.core.fromnumeric import argmax, argsort, shape
import torch
from torch._C import device
import torch.optim as optim
import model
import torch.nn as nn
import torch.utils.data as Data
from sklearn import metrics
import math
import copy
import MCC_loss
from time import time


def TrainNetwork(Model,labelled_data,labelled_target,test_data,test_target,lr_rate,wd_rate,loss_func,seed,device):
    
    num_labelled=labelled_data.shape[0]
    num_class=labelled_target.shape[0]

    #设置验证集
    num_test=test_data.shape[0]
    num_validation=round(num_test*0.2)
    validation_data=test_data[0:num_validation,:]
    validation_target=test_target[:,0:num_validation]


    X=torch.from_numpy(labelled_data).float().to(device)
    Y=torch.from_numpy(np.transpose(labelled_target)).float().to(device)

    Model.to(device)

    optimizer=optim.Adam(Model.parameters(),lr=10**(lr_rate-10), weight_decay=10**(wd_rate-10))

    #mini-batch with DataLoader
    BATCH_SIZE=32
    torch_dataset=Data.TensorDataset(X,Y)

    def worker_init_fn_seed(worker_id,seed):
        seed+=worker_id
        np.random.seed(seed)

    loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init_fn_seed
    )

    #保存初始模型
    torch.save(Model.state_dict(),'TorchModel.pth')
    best_micorF1=0
    stop_criterion=0
    max_iteration=100

    for epoch in range(max_iteration):
        for step, (batch_x, batch_y) in enumerate(loader):

            Model.train()
            optimizer.zero_grad()
            out=Model.forward(batch_x)
            loss=loss_func(out,batch_y)

            loss.backward()
            optimizer.step()
        with torch.no_grad():
            #在验证集上测试效果
            Model.eval()
            Model.to(device)
            Outputs=Model(torch.from_numpy(validation_data).float().to(device))
            Outputs=Outputs.cpu().detach().numpy()
            PreLabels=1*(Outputs>0.5)
            for i in range(0,num_validation):
                if np.max(PreLabels[i,:])==0:
                    PreLabels[i,np.argmax(Outputs[i,:])]=1
                if np.min(PreLabels[i,:])==1:
                    PreLabels[i,np.argmin(Outputs[i,:])]=0
            microF1=metrics.f1_score(np.transpose(validation_target),PreLabels,average='micro')


            if microF1>best_micorF1:
                # print(microF1)
                torch.save(Model.state_dict(),'TorchModel.pth')
                stop_criterion=0
                best_micorF1=microF1
            else:
                # print(microF1)
                stop_criterion=stop_criterion+1
            
            if stop_criterion>20:
                Model.load_state_dict(torch.load('TorchModel.pth'))
                break
    
    if epoch==max_iteration:
        Model.load_state_dict(torch.load('TorchModel.pth'))
    # print('epoch=',epoch)
    return Model

def TestPerformance(Model,test_data,test_target,lr_rate,wd_rate,loss_func,device):

    with torch.no_grad():
        Model.eval()
        Model.to(device)
        Outputs=Model(torch.from_numpy(test_data).float().to(device))
        Outputs=Outputs.cpu().detach().numpy()
    
    rankingloss=ranking_loss(np.transpose(test_target),Outputs)
    coverage=metrics.coverage_error(np.transpose(test_target),Outputs)
    coverage=coverage/test_target.shape[0]
    average_precision=metrics.average_precision_score(np.transpose(test_target),Outputs,average='samples')

    #evaluation metrics for binary value
    threshold=0.5
    PreLabels=1*(Outputs>threshold)

    for i in range(0,test_data.shape[0]):
        if np.max(PreLabels[i,:])==0:
            PreLabels[i,np.argmax(Outputs[i,:])]=1
        if np.min(PreLabels[i,:])==1:
            PreLabels[i,np.argmin(Outputs[i,:])]=0
    
    hammingloss=metrics.hamming_loss(np.transpose(test_target),PreLabels)
    microF1=metrics.f1_score(np.transpose(test_target),PreLabels,average='micro')

    return [hammingloss,rankingloss,coverage,average_precision,microF1]

def ranking_loss(y_true, y_pred_value):
    rloss = 0
    for i in range(y_true.shape[0]):
        invalid_pair_cnt = 0
        y_true_i = y_true[i, :]
        y_value_i = y_pred_value[i, :]
        pos = np.where(y_true_i == 1)[0]
        neg = np.where(y_true_i == 0)[0]
        for j in pos:
            for k in neg:
                if y_value_i[j] <= y_value_i[k]:
                    invalid_pair_cnt += 1
        if len(pos) * len(neg) != 0:
            rloss += float(invalid_pair_cnt) / (len(pos) * len(neg))
    return rloss / y_true.shape[0]

def QueryStrategyRandom(labelled_data, labelled_target, unlabelled_data, unlabelled_target, loss_func, device):

    chosen_script=0

    chosen_sample = unlabelled_data[chosen_script]
    chosen_target = unlabelled_target[:,chosen_script]

    # labelled_data = np.insert(labelled_data, labelled_data.shape[0], unlabelled_data[chosen_script], axis=0)
    # labelled_target = np.insert(labelled_target,labelled_target.shape[1],unlabelled_target[:,chosen_script],axis=1)

    unlabelled_data = np.delete(unlabelled_data, chosen_script, axis=0)
    unlabelled_target = np.delete(unlabelled_target, chosen_script, axis=1)

    return [chosen_sample,chosen_target,labelled_data,labelled_target,unlabelled_data,unlabelled_target]

def QueryStrategyMMU(Model, labelled_data, labelled_target, unlabelled_data, unlabelled_target, loss_func, device):
    
    num_labelled=labelled_data.shape[0]
    num_unlabelled=unlabelled_data.shape[0]

    #计算未标记数据的预测概率
    with torch.no_grad():
        Model.eval()
        Model.to(device)
        Outputs_unlabelled=Model(torch.from_numpy(unlabelled_data).float().to(device))
        Outputs_unlabelled=Outputs_unlabelled.cpu().detach().numpy()
    

    threshold=0.5
    PreLabels_unlabelled=1*(Outputs_unlabelled>threshold)

    for i in range(0,num_unlabelled):
        if np.max(PreLabels_unlabelled[i,:])==0:
            PreLabels_unlabelled[i,np.argmax(Outputs_unlabelled[i,:])]=1
        if np.min(PreLabels_unlabelled[i,:])==1:
            PreLabels_unlabelled[i,np.argmin(Outputs_unlabelled[i,:])]=0

    Outputs_unlabelled=np.transpose(Outputs_unlabelled)
    PreLabels_unlabelled=np.transpose(PreLabels_unlabelled)
    #NMMU算法
    MaxMarginUncertainty=np.zeros(num_unlabelled)
    for i in range(0,num_unlabelled):
        positive_index=np.where(PreLabels_unlabelled[:,i]==1)
        negative_index=np.where(PreLabels_unlabelled[:,i]==0)
        positivemin=np.min(Outputs_unlabelled[positive_index,i])
        negativemax=np.max(Outputs_unlabelled[negative_index,i])
        MaxMarginUncertainty[i]=1/(positivemin-negativemax)
    chosen_script=np.argmax(MaxMarginUncertainty)

    chosen_sample = unlabelled_data[chosen_script]
    chosen_target = unlabelled_target[:,chosen_script]

    # labelled_data = np.insert(labelled_data, labelled_data.shape[0], unlabelled_data[chosen_script], axis=0)
    # labelled_target = np.insert(labelled_target,labelled_target.shape[1],unlabelled_target[:,chosen_script],axis=1)

    unlabelled_data = np.delete(unlabelled_data, chosen_script, axis=0)
    unlabelled_target = np.delete(unlabelled_target, chosen_script, axis=1)


    return [chosen_sample,chosen_target,labelled_data,labelled_target,unlabelled_data,unlabelled_target]

def QueryStrategyLCI(Model, labelled_data, labelled_target, unlabelled_data, unlabelled_target, loss_func, device):

    num_labelled=labelled_data.shape[0]
    num_unlabelled=unlabelled_data.shape[0]

    #计算未标记数据的预测概率
    with torch.no_grad():
        Model.eval()
        Model.to(device)
        Outputs_unlabelled=Model(torch.from_numpy(unlabelled_data).float().to(device))
        Outputs_unlabelled=Outputs_unlabelled.cpu().detach().numpy()
    

    threshold=0.5
    PreLabels_unlabelled=1*(Outputs_unlabelled>threshold)

    for i in range(0,num_unlabelled):
        if np.max(PreLabels_unlabelled[i,:])==0:
            PreLabels_unlabelled[i,np.argmax(Outputs_unlabelled[i,:])]=1
        if np.min(PreLabels_unlabelled[i,:])==1:
            PreLabels_unlabelled[i,np.argmin(Outputs_unlabelled[i,:])]=0

    Outputs_unlabelled=np.transpose(Outputs_unlabelled)
    PreLabels_unlabelled=np.transpose(PreLabels_unlabelled)

    #实现LCI算法
    Cardinality_Uncertainty=np.zeros(num_unlabelled)
    Cardinality_Labelled=np.sum(labelled_target)/num_unlabelled
    for i in range(0,num_unlabelled):
        Cardinality_Uncertainty[i]=(np.sum(PreLabels_unlabelled[:,i])-Cardinality_Labelled)**2
    
    chosen_script=np.argmax(Cardinality_Uncertainty)

    chosen_sample = unlabelled_data[chosen_script]
    chosen_target = unlabelled_target[:,chosen_script]

    # labelled_data = np.insert(labelled_data, labelled_data.shape[0], unlabelled_data[chosen_script], axis=0)
    # labelled_target = np.insert(labelled_target,labelled_target.shape[1],unlabelled_target[:,chosen_script],axis=1)

    unlabelled_data = np.delete(unlabelled_data, chosen_script, axis=0)
    unlabelled_target = np.delete(unlabelled_target, chosen_script, axis=1)

    return [chosen_sample,chosen_target,labelled_data,labelled_target,unlabelled_data,unlabelled_target]

def QueryStrategyCVIRS(Model, labelled_data, labelled_target, unlabelled_data, unlabelled_target, loss_func, device):
    # #基于已标记数据经验损失和未标记数据预测概率的选择策略
    num_labelled=labelled_data.shape[0]
    num_unlabelled=unlabelled_data.shape[0]
    num_class=labelled_target.shape[0]

    if num_unlabelled==1:
        chosen_script=0

        chosen_sample = unlabelled_data[chosen_script]
        chosen_target = unlabelled_target[:,chosen_script]

        # labelled_data = np.insert(labelled_data, labelled_data.shape[0], unlabelled_data[chosen_script], axis=0)
        # labelled_target = np.insert(labelled_target,labelled_target.shape[1],unlabelled_target[:,chosen_script],axis=1)

        unlabelled_data = np.delete(unlabelled_data, chosen_script, axis=0)
        unlabelled_target = np.delete(unlabelled_target, chosen_script, axis=1)
        
        return [chosen_sample, chosen_target, labelled_data, labelled_target, unlabelled_data, unlabelled_target]
    
    #计算未标记数据的预测概率
    with torch.no_grad():
        Model.eval()
        Model.to(device)
        Prediction_unlabelled=Model(torch.from_numpy(unlabelled_data).float().to(device))
        Prediction_unlabelled=Prediction_unlabelled.cpu().detach().numpy()
    

    threshold=0.5
    PreLabels_unlabelled=1*(Prediction_unlabelled>threshold)

    for i in range(0,num_unlabelled):
        if np.max(PreLabels_unlabelled[i,:])==0:
            PreLabels_unlabelled[i,np.argmax(Prediction_unlabelled[i,:])]=1
        if np.min(PreLabels_unlabelled[i,:])==1:
            PreLabels_unlabelled[i,np.argmin(Prediction_unlabelled[i,:])]=0

    Prediction_unlabelled=np.transpose(Prediction_unlabelled)
    PreLabels_unlabelled=np.transpose(PreLabels_unlabelled)
    
    #计算第一个指标，rank aggregation problem
    Positive_Possibility=Prediction_unlabelled
    Negative_Possibility=1-Prediction_unlabelled
    Margin=np.abs(Positive_Possibility-Negative_Possibility)

    Order=np.zeros(shape=(num_class,num_unlabelled))
    for i in range(0,num_class):
        temp_index=np.argsort(Margin[i,:])
        for j in range(0,num_unlabelled):
            Order[i,temp_index[j]]=j+1
    Uncertainty_S=np.zeros(num_unlabelled)
    for j in range(0,num_unlabelled):
        temp_uncertain=0
        for i in range(0,num_class):
            temp_uncertain=temp_uncertain+(num_unlabelled-Order[i,j])
        temp_uncertain=temp_uncertain/(num_class*(num_unlabelled-1))
        Uncertainty_S[j]=temp_uncertain
    
    #计算第二个指标，entropy like inconsistency
    Hamming_dis=np.zeros(shape=(num_unlabelled,num_labelled))
    Entropy_dis=np.zeros(shape=(num_unlabelled,num_labelled))
    for i in range(0,num_unlabelled):
        for j in range(0,num_labelled):
            Y_unlabelled=PreLabels_unlabelled[:,i]
            Y_labelled=labelled_target[:,j]

            a=np.sum((Y_unlabelled*Y_labelled)==1)
            d=np.sum((Y_unlabelled+Y_labelled)==0)
            b=np.sum(Y_unlabelled==1)-a
            c=np.sum(Y_labelled==1)-a

            Hamming_dis[i,j]=(b+c)/num_class
            if b+c==0:
                Hij=H2((b+c)/num_class,(a+d)/num_class)+((a+d)/num_class)*H2(a/(a+d),d/(a+d))
            else:
                if a+d==0:
                    Hij=H2((b+c)/num_class,(a+d)/num_class)+((b+c)/num_class)*H2(b/(b+c),c/(b+c))
                else:
                    Hij=H2((b+c)/num_class,(a+d)/num_class)+((b+c)/num_class)*H2(b/(b+c),c/(b+c))+((a+d)/num_class)*H2(a/(a+d),d/(a+d))
    
            Hi=H2(sum(Y_unlabelled)/num_class,1-sum(Y_unlabelled)/num_class)
            Hj=H2(sum(Y_labelled)/num_class,1-sum(Y_labelled)/num_class)
            Entropy_dis[i,j]=(2*Hij-Hi-Hj)/Hij

    Uncertainty_V=np.zeros(num_unlabelled)
    for i in range(0,num_unlabelled):
        temp_uncertain=0
        for j in range(0,num_labelled):
            if Hamming_dis[i,j]==1:
                temp_uncertain=temp_uncertain+1
            else:
                temp_uncertain=temp_uncertain+Entropy_dis[i,j]
        Uncertainty_V[i]=temp_uncertain/num_labelled
    
    #两个指标组合
    Uncertainty=Uncertainty_S*Uncertainty_V
    chosen_script=np.argmax(Uncertainty)

    chosen_sample = unlabelled_data[chosen_script]
    chosen_target = unlabelled_target[:,chosen_script]

    # labelled_data = np.insert(labelled_data, labelled_data.shape[0], unlabelled_data[chosen_script], axis=0)
    # labelled_target = np.insert(labelled_target,labelled_target.shape[1],unlabelled_target[:,chosen_script],axis=1)

    unlabelled_data = np.delete(unlabelled_data, chosen_script, axis=0)
    unlabelled_target = np.delete(unlabelled_target, chosen_script, axis=1)
    
    return [chosen_sample, chosen_target, labelled_data, labelled_target, unlabelled_data, unlabelled_target]

def H2(w,s):
    if w*s==0:
        entropy=0
    else:
        entropy=-w*np.log2(w)-s*np.log2(s)
    return entropy

def QueryStrategyRMLAL(Model,labelled_data,labelled_target,unlabelled_data,unlabelled_target,DisFea,labelled_index,unlabelled_index,loss_func,device):
    
    num_labelled=labelled_data.shape[0]
    num_unlabelled=unlabelled_data.shape[0]
    num_class=labelled_target.shape[0]
    num_fea=labelled_data.shape[1]

    #计算未标记数据的预测概率
    with torch.no_grad():
        Model.eval()
        Model.to(device)
        Outputs=Model(torch.from_numpy(unlabelled_data).float().to(device))
        Outputs=Outputs.cpu().detach().numpy()
    #计算未标记数据的预测标记
    threshold=0.5
    PreLabels=1*(Outputs>threshold)
    #防止出现预测label全0或者全1的现象
    for i in range(num_unlabelled):
        if np.max(PreLabels[i,:])==0:
            PreLabels[i,np.argmax(Outputs[i,:])]=1
        if np.min(PreLabels[i,:])==1:
            PreLabels[i,np.argmin(Outputs[i,:])]=0
    
    Outputs=np.transpose(Outputs)
    PreLabels=np.transpose(PreLabels)

    #将预测概率、预测标记和真实标记变换到[-1,1]区间
    Outputs=(Outputs-0.5)*2
    PreLabels=(PreLabels-0.5)*2
    labelled_target1=(labelled_target-0.5)*2


    #实现RMAL算法
    sigma_label=1/num_class
    sigma_feature=1/num_fea
    beta1=1
    beta2=1

    #计算Informative
    Informative=np.zeros(num_unlabelled)
    for i in range(num_class):
        temp_info=Outputs[i,:]
        temp_info=(1+2*abs(temp_info)+temp_info**2)
        for j in range(num_unlabelled):
            temp_info[j]=math.exp(temp_info[j]*(-sigma_label))
        Informative=Informative+temp_info
    
    #计算DisPreUU
    DisPreUU=np.zeros(shape=(num_unlabelled,num_unlabelled))
    for i in range(num_unlabelled):
        for j in range(num_unlabelled):
            DisPreUU[i,j]=sum((PreLabels[:,i]-PreLabels[:,j])**2)
            DisPreUU[j,i]=DisPreUU[i,j]
    #计算DisPreUL
    DisPreUL=np.zeros(shape=(num_unlabelled,num_labelled))
    for i in range(num_unlabelled):
        for j in range(num_labelled):
            DisPreUL[i,j]=sum((PreLabels[:,i]-labelled_target1[:,j])**2)
    
    #计算Rep1
    Rep1=np.zeros(num_unlabelled)
    for i in range(num_unlabelled):
        temp_rep1=0
        for j in range(num_unlabelled):
            temp_rep1=temp_rep1+math.exp(-sigma_label*DisPreUU[i,j])*DisFea[unlabelled_index[i],unlabelled_index[j]]
        Rep1[i]=beta1*(1/num_unlabelled)*temp_rep1
    #计算Rep2
    Rep2=np.zeros(num_unlabelled)
    for i in range(num_unlabelled):
        temp_rep2=0
        for j in range(num_labelled):
            temp_rep2=temp_rep2+math.exp(-sigma_label*DisPreUL[i,j])*DisFea[unlabelled_index[i],labelled_index[j]]
        Rep2[i]=beta2*(1/num_labelled)*temp_rep2
    
    #计算最终得分，并排序找到得分最大值
    score=Informative+Rep1-Rep2
    chosen_script=np.argmax(score)

    chosen_sample = unlabelled_data[chosen_script]
    chosen_target = unlabelled_target[:,chosen_script]

    unlabelled_data = np.delete(unlabelled_data, chosen_script, axis=0)
    unlabelled_target = np.delete(unlabelled_target, chosen_script, axis=1)

    #更新索引
    labelled_index=np.append(labelled_index,unlabelled_index[chosen_script])
    unlabelled_index=np.delete(unlabelled_index,chosen_script)

    return [chosen_sample,chosen_target,labelled_data,labelled_target,unlabelled_data,unlabelled_target,labelled_index,unlabelled_index]


def ActiveLearning(labelled_data,labelled_target,unlabelled_data,unlabelled_target,test_data,test_target,lr_rate,wd_rate,query_mode,mistake_mode,num_mistake,difficult_threshold,seed):
    
    #数据集参数
    num_class=np.size(labelled_target,0)
    num_fea=np.size(labelled_data,1)
    num_labelled_ins=np.size(labelled_data,0)
    num_unlabelled_ins=np.size(unlabelled_data,0)
    num_test_ins=np.size(test_data,0)


    #计算RMLAL算法所需的样本距离矩阵
    if query_mode==5:
        train_data=np.vstack((labelled_data,unlabelled_data))
        num_train=num_labelled_ins+num_unlabelled_ins

        labelled_index=np.array(range(num_labelled_ins))
        unlabelled_index=np.array(range(num_labelled_ins,num_train))

        DisFea=np.zeros(shape=(num_train,num_train))
        for i in range(num_train):
            # print('i=',i)
            for j in range(num_train):
                DisFea[i,j]=sum((train_data[i,:]-train_data[j,:])**2)
                DisFea[i,j]=math.exp(-(1/num_fea)*DisFea[i,j])
                DisFea[j,i]=DisFea[i,j]
    


    #设置CUDA加速
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #定义神经网络并首次训练
    Model=model.Net(num_fea,num_class,seed)
    if query_mode==5:
        loss_func=MCC_loss.CorrentropyLoss()
    else:
        loss_func=nn.BCELoss()
    
    # loss_func=nn.BCELoss()
    
    Model=TrainNetwork(Model,labelled_data,labelled_target,test_data,test_target,lr_rate,wd_rate,loss_func,seed,device)
    [hammingloss,rankingloss,coverage,average_precision,microF1]=TestPerformance(Model,test_data,test_target,lr_rate,wd_rate,loss_func,device)
    print('hammingloss=',hammingloss,'rankingloss=',rankingloss,'average_precision=',average_precision,'microF1=',microF1)

    #主动学习策略
    selection_round=0
    sum_mistake=0

    HammingLoss=np.array([hammingloss])
    RankingLoss=np.array([rankingloss])
    Coverage=np.array([coverage])
    Average_Precision=np.array([average_precision])
    MicroF1=np.array([microF1])
    AnnotationCost=np.array([0])
    MistakeRecord=np.array([sum_mistake])
    TargetDiffer=np.zeros(num_class)

    while unlabelled_data.size != 0:
        selection_round=selection_round+1
        print('selection_round:',selection_round)

        #选择主动学习的策略
        if query_mode==1:
            [chosen_sample, chosen_target, labelled_data, labelled_target, unlabelled_data, unlabelled_target] = QueryStrategyRandom(labelled_data, labelled_target, unlabelled_data, unlabelled_target, loss_func, device)
        elif query_mode==2:
            [chosen_sample, chosen_target, labelled_data, labelled_target, unlabelled_data, unlabelled_target] = QueryStrategyMMU(Model,labelled_data, labelled_target, unlabelled_data, unlabelled_target, loss_func, device)
        elif query_mode==3:
            [chosen_sample, chosen_target, labelled_data, labelled_target, unlabelled_data, unlabelled_target] = QueryStrategyLCI(Model,labelled_data, labelled_target, unlabelled_data, unlabelled_target, loss_func, device)
        elif query_mode==4:
            [chosen_sample, chosen_target, labelled_data, labelled_target, unlabelled_data, unlabelled_target] = QueryStrategyCVIRS(Model,labelled_data, labelled_target, unlabelled_data, unlabelled_target, loss_func, device)
        elif query_mode==5:
            [chosen_sample,chosen_target,labelled_data,labelled_target,unlabelled_data,unlabelled_target,labelled_index,unlabelled_index]=QueryStrategyRMLAL(Model,labelled_data,labelled_target,unlabelled_data,unlabelled_target,DisFea,labelled_index,unlabelled_index,loss_func,device)
        
        #模拟标记错误的情况
        [labelled_data,labelled_target,whether_mistake,targetdiffer]=AnnotationMistake(Model,labelled_data,labelled_target,chosen_sample,chosen_target,mistake_mode,num_mistake,difficult_threshold,device)
        
        sum_mistake=sum_mistake+whether_mistake
        MistakeRecord=np.append(MistakeRecord,sum_mistake)
        TargetDiffer=TargetDiffer+targetdiffer

        if selection_round%2==0:

            #重新训练神经网络
            # Model=model.Net(num_fea,num_class,seed)
            Model=TrainNetwork(Model,labelled_data,labelled_target,test_data,test_target,lr_rate,wd_rate,loss_func,seed,device)

            #在测试集上计算效果
            [hammingloss,rankingloss,coverage,average_precision,microF1]=TestPerformance(Model,test_data,test_target,lr_rate,wd_rate,loss_func,device)
            print('hammingloss=',hammingloss,'rankingloss=',rankingloss,'average_precision=',average_precision,'microF1=',microF1)
            
            HammingLoss=np.append(HammingLoss,hammingloss)
            RankingLoss=np.append(RankingLoss, rankingloss)
            Coverage=np.append(Coverage,coverage)
            Average_Precision=np.append(Average_Precision,average_precision)
            MicroF1=np.append(MicroF1,microF1)
            AnnotationCost=np.append(AnnotationCost,selection_round*num_class)
    
    #测试最终效果
    # print('selection end')
    # Model=TrainNetwork(Model,labelled_data,labelled_target,test_data,test_target,lr_rate,wd_rate,loss_func,seed,device)
    # [hammingloss,rankingloss,coverage,average_precision,microF1]=TestPerformance(Model,test_data,test_target,lr_rate,wd_rate,loss_func,device)
    # print('hammingloss=',hammingloss,'rankingloss=',rankingloss,'average_precision=',average_precision,'microF1=',microF1)



    return [Model,HammingLoss,RankingLoss,Coverage,Average_Precision,MicroF1,AnnotationCost,MistakeRecord,TargetDiffer]

def AnnotationMistake(Model,labelled_data,labelled_target,chosen_sample,chosen_target,mistake_mode,num_mistake,difficult_threshold,device):

    if mistake_mode==0:
        labelled_data = np.insert(labelled_data, labelled_data.shape[0], chosen_sample, axis=0)
        labelled_target = np.insert(labelled_target,labelled_target.shape[1],chosen_target,axis=1)
        whether_mistake=0
        targetdiffer=np.zeros(labelled_target.shape[0])
        return [labelled_data,labelled_target,whether_mistake,targetdiffer]

    num_class=labelled_target.shape[0]

    # num_mistake=1

    mistake_target=copy.deepcopy(chosen_target)

    with torch.no_grad():
        Model.eval()
        Model.to(device)
        p=Model(torch.from_numpy(chosen_sample).float().to(device))
        p=p.cpu().detach().numpy()
    
    ascend_index=np.argsort(p)
    descend_index=np.argsort(-p)

    if mistake_mode==1: #difficult positive label
        p_index=ascend_index
        true_value=1
        false_value=0
        ordered_target=chosen_target[p_index]
        ordered_target=ordered_target[0:round(num_class*(1-difficult_threshold))]
    elif mistake_mode==2: #easy positive label
        p_index=descend_index
        true_value=1
        false_value=0
        ordered_target=chosen_target[p_index]
        ordered_target=ordered_target[0:round(num_class*difficult_threshold)]
    elif mistake_mode==3: #difficult negative label
        p_index=descend_index
        true_value=0
        false_value=1
        ordered_target=chosen_target[p_index]
        ordered_target=ordered_target[0:round(num_class*difficult_threshold)]
    else: #easy negative label
        p_index=ascend_index
        true_value=0
        false_value=1
        ordered_target=chosen_target[p_index]
        ordered_target=ordered_target[0:round(num_class*(1-difficult_threshold))]
    

    temp=list(np.where(ordered_target==true_value))
    temp=temp[0]
    if len(temp)==0:
        whether_mistake=0
    else:
        temp=temp[0:min(num_mistake,len(temp))]
        mistake_target[p_index[temp]]=false_value
        whether_mistake=len(temp)

    labelled_data = np.insert(labelled_data, labelled_data.shape[0], chosen_sample, axis=0)
    labelled_target = np.insert(labelled_target,labelled_target.shape[1],mistake_target,axis=1)


    #统计被标注错误的label
    targetdiffer=np.zeros(num_class)
    for i in range(num_class):
        if mistake_target[i]!=chosen_target[i]:
            targetdiffer[i]=1

    return [labelled_data,labelled_target,whether_mistake,targetdiffer]