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
import ExaLab_loss


#训练一个多标记神经网络，梯度下降的停止条件是验证集上的效果在10次epoch中没有任何提升，最后返回效果最佳时的模型。
def TrainNetwork(Model,labelled_data,labelled_target,test_data,test_target,IndicateMatirx,labelled_index,lr_rate,wd_rate,loss_func,seed,device):

    num_labelled=labelled_data.shape[0]
    num_class=labelled_target.shape[0]

    #设置验证集，验证集即为run.py中的test_data
    num_test=test_data.shape[0]
    num_validation=round(num_test*0.2)
    validation_data=test_data[0:num_validation,:]
    validation_target=test_target[:,0:num_validation]


    #labelled_data中前面一部分是完全已标注样本，后面一部分是部分已标注样本
    # copy_labelled_target用于指示labelled_data中哪些label已经标注，没有标注的部分不参与loss的计算
    copy_labelled_target=copy.deepcopy(labelled_target)

    for i in range(num_labelled):
        for j in range(num_class):
            if IndicateMatirx[j,labelled_index[i]]==0:
                copy_labelled_target[j,i]=2

    X=torch.from_numpy(labelled_data).float().to(device)
    Y=torch.from_numpy(np.transpose(copy_labelled_target)).float().to(device)

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
            # print('out=',out)
            loss=loss_func(out,batch_y)
            # print('loss=',loss)

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

            # print('microF1=',microF1)

            if microF1>best_micorF1:
                torch.save(Model.state_dict(),'TorchModel.pth')
                stop_criterion=0
                best_micorF1=microF1
            else:
                stop_criterion=stop_criterion+1
            
            # 如果连续10次在验证集上的效果没有提升则停止，模型回到最佳效果时的状态
            if stop_criterion>10:
                Model.load_state_dict(torch.load('TorchModel.pth'))
                break
    
    if epoch==max_iteration:
        Model.load_state_dict(torch.load('TorchModel.pth'))
    # print('epoch=',epoch)
    return Model

#利用输入的模型输出测试集上的效果，包括microF1 hammingloss rankingloss coverage average_precision
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

#计算ranking_loss，在函数TestPerformance中被调用
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



    chosen_script=0

    chosen_sample = unlabelled_data[chosen_script]
    chosen_target = unlabelled_target[:,chosen_script]

    # labelled_data = np.insert(labelled_data, labelled_data.shape[0], unlabelled_data[chosen_script], axis=0)
    # labelled_target = np.insert(labelled_target,labelled_target.shape[1],unlabelled_target[:,chosen_script],axis=1)

    unlabelled_data = np.delete(unlabelled_data, chosen_script, axis=0)
    unlabelled_target = np.delete(unlabelled_target, chosen_script, axis=1)

    return [chosen_sample,chosen_target,labelled_data,labelled_target,unlabelled_data,unlabelled_target]

#AUDI(instance-labe pair based)算法的query strategy，每次选择一个被认为最有价值的instance-label pair
def QueryStrategyAUDI(Model,labelled_data,labelled_target,unlabelled_data,unlabelled_target,IndicateMatirx,labelled_index,unlabelled_index,loss_func,device):

    #数据参数
    num_labelled=labelled_data.shape[0]
    num_unlabelled=unlabelled_data.shape[0]
    num_class=labelled_target.shape[0]

    #合并已标记数据和未标记数据
    if num_unlabelled != 0:
        train_data=np.vstack((labelled_data,unlabelled_data))
        train_index=np.hstack((labelled_index,unlabelled_index))
    else:
        train_data=labelled_data
        train_index=labelled_index
    
    num_train=train_data.shape[0]
    

    #计算每个instance的预测概率
    with torch.no_grad():
        Model.eval()
        Model.to(device)
        Outputs=Model(torch.from_numpy(train_data).float().to(device))
        Outputs=Outputs.cpu().detach().numpy()
    threshold=0.5
    PreLabels=1*(Outputs>threshold)
    Outputs=np.transpose(Outputs)
    PreLabels=np.transpose(PreLabels)


    #计算完全已标注样本的cardinality
    cardinality=0
    count=0
    for i in range(num_labelled):
        if min(IndicateMatirx[:,labelled_index[i]])==1:
            cardinality=cardinality+sum(labelled_target[:,i])
            count=count+1
    cardinality=cardinality/count

    #query一个最有价值的instance
    score=np.zeros(num_train)
    for i in range(num_train):
        if min(IndicateMatirx[:,train_index[i]])==1:
            continue
        else:
            temp_score=sum(PreLabels[:,i])
            temp_score=abs(temp_score-cardinality)/max(0.5,sum(IndicateMatirx[:,train_index[i]]))
            score[i]=temp_score

    #query最有价值的instance的最有价值label
    chosen_instance=np.argmax(score)
    chosen_label=0
    min_dis=1
    for i in range(num_class):
        if IndicateMatirx[i,train_index[chosen_instance]]==0:
            temp_dis=abs(Outputs[i,chosen_instance]-0.5)
            if temp_dis<min_dis:
                chosen_label=i
                min_dis=temp_dis
    

    #更新IndicateMatrix等, IndicateMatrix记录有哪些instance-label已经标注

    #如果被选中的这个instance已经有label被标注过
    if chosen_instance<num_labelled:
        IndicateMatirx[chosen_label,train_index[chosen_instance]]=1
    #如果这个instance完全没有被标注过
    else:
        IndicateMatirx[chosen_label,train_index[chosen_instance]]=1
        chosen_script=chosen_instance-num_labelled
            
        labelled_data = np.insert(labelled_data, labelled_data.shape[0], unlabelled_data[chosen_script], axis=0)
        labelled_target = np.insert(labelled_target,labelled_target.shape[1],unlabelled_target[:,chosen_script],axis=1)

        unlabelled_data = np.delete(unlabelled_data, chosen_script, axis=0)
        unlabelled_target = np.delete(unlabelled_target, chosen_script, axis=1)

        if num_unlabelled != 0:
            labelled_index=np.append(labelled_index,unlabelled_index[chosen_script])
            unlabelled_index=np.delete(unlabelled_index,chosen_script)


    return [labelled_data,labelled_target,unlabelled_data,unlabelled_target,IndicateMatirx,labelled_index,unlabelled_index]



def ActiveLearning(labelled_data,labelled_target,unlabelled_data,unlabelled_target,test_data,test_target,lr_rate,wd_rate,num_mistake,difficult_threshold,seed):
    
    #数据集参数
    num_class=np.size(labelled_target,0)
    num_fea=np.size(labelled_data,1)
    num_labelled_ins=np.size(labelled_data,0)
    num_unlabelled_ins=np.size(unlabelled_data,0)
    num_test_ins=np.size(test_data,0)
    num_train=num_labelled_ins+num_unlabelled_ins

    #设置CUDA加速
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #设置一个指示矩阵，标明各个example-label pair是否被标注。初始时已标注集所有label都被标明已标注。
    IndicateMatirx=np.zeros(shape=(num_class,num_train))
    for i in range(num_labelled_ins):
        IndicateMatirx[:,i]=IndicateMatirx[:,i]+1
    #索引向量
    labelled_index=np.array(range(num_labelled_ins))
    unlabelled_index=np.array(range(num_labelled_ins,num_train))


    #定义神经网络并首次训练
    Model=model.Net(num_fea,num_class,seed)
    #自定义的适合instance-label pair的二类交叉熵损失函数
    loss_func=ExaLab_loss.ExaLabBCELoss()
    
    Model=TrainNetwork(Model,labelled_data,labelled_target,test_data,test_target,IndicateMatirx,labelled_index,lr_rate,wd_rate,loss_func,seed,device)

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

    while np.sum(IndicateMatirx) != (num_class*num_train):
        selection_round=selection_round+1
        if selection_round%num_class==0:
            print('selection_round:',selection_round/num_class)

        #AUDI的query strategy
        [labelled_data,labelled_target,unlabelled_data,unlabelled_target,IndicateMatirx,labelled_index,unlabelled_index] = QueryStrategyAUDI(Model,labelled_data,labelled_target,unlabelled_data,unlabelled_target,IndicateMatirx,labelled_index,unlabelled_index,loss_func,device)

        # 每选择 2*num_class个label pair时，训练一次
        if selection_round%(2*num_class)==0:

            #在Query了2*num_class个label之后继续/重新训练神经网络。
            # Model=model.Net(num_fea,num_class,seed)
            Model=TrainNetwork(Model,labelled_data,labelled_target,test_data,test_target,IndicateMatirx,labelled_index,lr_rate,wd_rate,loss_func,seed,device)


            #在测试集上计算效果
            [hammingloss,rankingloss,coverage,average_precision,microF1]=TestPerformance(Model,test_data,test_target,lr_rate,wd_rate,loss_func,device)
            print('hammingloss=',hammingloss,'rankingloss=',rankingloss,'average_precision=',average_precision,'microF1=',microF1)
            
            HammingLoss=np.append(HammingLoss,hammingloss)
            RankingLoss=np.append(RankingLoss, rankingloss)
            Coverage=np.append(Coverage,coverage)
            Average_Precision=np.append(Average_Precision,average_precision)
            MicroF1=np.append(MicroF1,microF1)
            AnnotationCost=np.append(AnnotationCost,selection_round)

    #测试最终效果
    # print('selection end')
    # loss_func=nn.BCELoss()
    # Model=TrainNetwork(Model,labelled_data,labelled_target,test_data,test_target,IndicateMatirx,labelled_index,lr_rate,wd_rate,loss_func,seed,device)
    # [hammingloss,rankingloss,coverage,average_precision,microF1]=TestPerformance(Model,test_data,test_target,lr_rate,wd_rate,loss_func,device)
    # print('hammingloss=',hammingloss,'rankingloss=',rankingloss,'average_precision=',average_precision,'microF1=',microF1)

    return [Model,HammingLoss,RankingLoss,Coverage,Average_Precision,MicroF1,AnnotationCost,MistakeRecord,TargetDiffer]