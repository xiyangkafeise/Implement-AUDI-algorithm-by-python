import torch
import torch.nn as nn
import math

class CorrentropyLoss(nn.Module):
    def _init_(self):
        super(CorrentropyLoss,self).__init__()
    
    def forward(self,output,target):

        #将预测值和真实值变换到[-1,1]区间
        target=(target-0.5)*2
        output=(output-0.5)*2


        cost=0
        num_ins=target.shape[0]
        num_class=target.shape[1]
        sigma=1/num_class

        for i in range(num_ins):
            for j in range(num_class):
                temp_cost=torch.pow(output[i,j]-target[i,j],2)
                temp_cost=1-torch.exp(temp_cost*(-sigma))
                cost=cost+temp_cost
        
        cost=cost/(num_ins*num_class)

        return cost
