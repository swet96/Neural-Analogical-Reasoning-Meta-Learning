import torch.nn as nn
from torch import optim
import torch


def accuracy(Net,X_test,y_test,verbose=True):
    Net.eval()
    m = X_test.shape[0]
    y_pred = Net(X_test)
    predicted = torch.max(y_pred, 1)[1]
    correct = (predicted == y_test).float().sum().item()
    if verbose: print(correct,m)
    accuracy = correct/m
    Net.train()
    return accuracy
# utils.hide_toggle('Function: accuracy')


class MLP(nn.Module):
    def __init__(self,dims=[5,3,2],task='classification'):
        super(MLP,self).__init__()
        self.dims=dims
        self.n = len(self.dims)-1
        self.task=task
        self.layers=nn.ModuleList()
        # self.initialize_weights()

        for i in range(self.n-1):
            self.layers.append(nn.Linear(dims[i],dims[i+1]))
            self.layers.append(nn.Dropout(0.8))
            self.layers.append(nn.ReLU())
        if task=='classification': 
            self.layers.append(nn.Linear(dims[i+1],dims[i+2]))
            
            self.layers.append(nn.LogSoftmax(dim=1))
        elif task=='regression': 
            self.layers.append(nn.Linear(dims[i+1],dims[i+2]))
            self.layers.append(nn.Linear(dims[i+2],1))
        else: self.layers.append(nn.Linear(dims[i+1],dims[i+2]))
        # self.optimizer = optim.Adam(self.parameters(),lr=lr)
    def forward(self,x):
        for l in self.layers:
            x = l(x)
        return(x)
    def initialize_weights(self):
        print(list[self.children()])
        # for m in self.children():
        #     print(m)
            # if isinstance(m, nn.Linear):
            #     nn.init.kaiming_uniform_(m.weight)
            # if m.bias is not None:
            #     nn.init.constant_(m.bias,0)



class MLP_MAN(nn.Module):
    def __init__(self,dims=[5,3,2],task='classification'):
        super(MLP_MAN,self).__init__()
        self.dims=dims
        self.n = len(self.dims)-1
        self.task=task
        self.layers=nn.ModuleList()
        # self.initialize_weights()

        for i in range(self.n-1):
            self.layers.append(nn.Linear(dims[i],dims[i+1]))
            self.layers.append(nn.ReLU())
        if task=='classification': 
            self.layers.append(nn.Linear(dims[i+1],dims[i+2]))
            self.layers.append(nn.LogSoftmax(dim=1))
        elif task=='regression': 
            self.layers.append(nn.Linear(dims[i+1],dims[i+2]))
            self.layers.append(nn.Linear(dims[i+2],1))
        else: self.layers.append(nn.Linear(dims[i+1],dims[i+2]))
        # self.optimizer = optim.Adam(self.parameters(),lr=lr)
    def forward(self,x):
        for l in self.layers:
            x = l(x)
        return(x)

    # def initialize_weights(self):
    #     print(list[self.children()])
    #     # for m in self.children():
    #     #     print(m)
    #         # if isinstance(m, nn.Linear):
    #         #     nn.init.kaiming_uniform_(m.weight)
    #         # if m.bias is not None:
    #         #     nn.init.constant_(m.bias,0)

    

class CNN(nn.Module):
    def __init__(self,n_channels):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 4, 3),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3,2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, 2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3,2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3)
        )
        # self.optimizer = optim.Adam(self.parameters(),lr=lr)
        # self.optimizer = optim.SGD(self.parameters(),lr= 0.1, momentum=0.9, weight_decay= 5e-4)


    def forward(self,x):
        # print("before: ", x.shape)
        out = self.cnn(x)
        # print("after: ", out.shape)
        return out