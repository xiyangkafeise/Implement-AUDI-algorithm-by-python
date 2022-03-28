import torch
import torch.nn as nn

class ExaLabBCELoss(nn.Module):
    def _init_(self):
        super(ExaLabBCELoss,self).__init__()
    
    def forward(self,output,target):
        cost=0

        #数据参数
        num_ins=target.shape[0]
        num_class=target.shape[1]

        count=0
        #逐一计算交叉熵损失 -[ylog(p)+(1-y)log(1-p)]
        for i in range(num_ins):
            for j in range(num_class):
                if target[i,j] != 2:
                    count=count+1
                    if target[i,j]==1:
                        cost=cost-torch.log(output[i,j])

                    elif target[i,j]==0:
                        cost=cost-torch.log(1-output[i,j])
        cost=cost/count
        return cost
