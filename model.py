import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_fea, num_class, seed, parameter_momentum=0.1):
        super(Net,self).__init__()
        L0=num_fea
        L1=round(num_fea/3)
        L2=num_class

        torch.manual_seed(seed)

        self.L1=nn.Linear(L0,L1,bias=True)
        torch.nn.init.xavier_uniform_(self.L1.weight)
        torch.nn.init.zeros_(self.L1.bias)

        self.L2 = nn.Linear(L1, L2, bias=True)
        torch.nn.init.xavier_uniform_(self.L2.weight)
        torch.nn.init.zeros_(self.L2.bias)

    def forward(self,x):
        x=self.L1(x)
        x=torch.tanh(x)

        x=self.L2(x)
        x=torch.sigmoid(x)

        return x
        