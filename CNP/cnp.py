
import sys
sys.path.append("/home/sweta/sweta_NAR_paper")

import models
import torch
import torch.nn as nn
from dataset import AnalogicalReasoningDataset
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import random
import logging
import os.path as osp
import torchvision  #ADDED
from torchsummary import summary


device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
print("Accelerator found: ", device)


def set_seed(seed):
  random.seed(0) #0
  np.random.seed(0) #0
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

seed = 17
set_seed(seed)
print("seed used is: ", seed)
logger_filename = f"./logger_files_cnp" 
json_filename= f"./problems_10k.json"
nar_filename= f"./nar_classification_dataset_images_10k/images_large"

ways=2
task_count=2000
n_epochs=60
print("(Tasks are non-overlapping) Number of tasks used for meta train and meta test: %s, Number of epochs: %s"%(task_count, n_epochs))
logging.info("(Tasks are non-overlapping) Number of tasks used for meta train and meta test: %s, Number of epochs: %s"%(task_count, n_epochs))


meta_train_size1 = 0
meta_train_size2 =task_count
meta_test_size1 = task_count
meta_test_size2 = 2*task_count


# model = torchvision.models.vgg16(pretrained=True)
# # print(model)
# model.avgpool = nn.AdaptiveAvgPool2d((1,1))

# modules = list(model.children())[:-1]
# model = nn.Sequential(*modules)
# # print(model)
# for p in model.parameters():
#   p.requires_grad = False
# for c in list(model.children())[0][26:]:
#   for p in c.parameters():
#     p.requires_grad = True


#Getting a single task     
def get_task(ex,is_cnn):
  if is_cnn:
    X_train = torch.stack([(ex[1].examples[i].input - ex[1].examples[i].options[j]) for i in range(len(ex[1].examples)) for j in range(len(ex[1].examples[i].options))])
  else:
    X_train = torch.stack([(ex[1].examples[i].input - ex[1].examples[i].options[j]).flatten() for i in range(len(ex[1].examples)) for j in range(len(ex[1].examples[i].options))])

  y_train = [F.one_hot(torch.tensor([ex[1].examples[i].solution]), num_classes=4) for i in range(len(ex[1].examples))]
  y_train =torch.stack(y_train).flatten() 

  if is_cnn:
    X_test = torch.stack([(ex[1].query-ex[1].query_options[i]) for i in range(len(ex[1].query_options))  ])
  else:
    X_test = torch.stack([(ex[1].query-ex[1].query_options[i]).flatten() for i in range(len(ex[1].query_options))  ])

  y_test = F.one_hot(torch.tensor(ex[1].solution), num_classes=len(ex[1].query_options))

  d_train=(X_train, y_train)
  d_test=(X_test, y_test)
  return d_train, d_test



class CNP(nn.Module):
  def __init__(self, n_features=1, dims=[32,32], n_classes=2, n_channels=3, is_cnn= False): #data=(X_test,(X_train,y_train))
    super(CNP,self).__init__()

    self.n_features   = n_features
    self.n_classes    = n_classes
    self.is_cnn       = is_cnn
    if is_cnn:
      # self.cnn =models.CNN(n_channels = n_channels)
      self.cnn = self.get_vgg_model()
      dims= dims+[n_classes]
      self.mlp2 = models.MLP(dims=dims)
      print("The CNN used to get average class embedding: \n %s ", self.cnn)
      logging.info("The CNN used to get average class embedding: \n %s ", self.cnn)

      print("\n")
      print("The MLP used to classify: \n %s ", self.mlp2)
      logging.info("The MLP used to classify: \n %s ", self.mlp2)


    else:
      dimL1             = [n_features]+dims
      self.mlp1         = models.MLP(dims=dimL1,task='embedding')
      dimL2             = [n_features+n_classes*dims[-1]]+dims+[n_classes] #TODO
      # dimL2             = [dims[-1]+n_classes*dims[-1]]+dims+[n_classes]
      self.mlp2         = models.MLP(dims=dimL2, task= 'classification')
      print("The MLP used to get average class embedding:: \n %s ", self.mlp1)
      logging.info("The MLP used to get average class embedding:: \n %s ", self.mlp1)

      print("\n")
      print("The MLP used to classify: \n %s ", self.mlp2)
      logging.info("The MLP used to classify: \n %s ", self.mlp2)



  # def get_vgg_model(self):
  #   model = torchvision.models.vgg11(pretrained=True)
  #   # print(model)
  #   model.avgpool = nn.AdaptiveAvgPool2d((1,1))

  #   modules = list(model.children())[:-1]
  #   # modules =[list(model.children())[0][:11], list(model.children())[1]]
  #   model = nn.Sequential(*modules)
  #   # print(model)
  #   for p in model.parameters():
  #     p.requires_grad = False
  #   for c in list(model.children())[0][14:]:
  #     for p in c.parameters():
  #       p.requires_grad = True
  #   return model

  def get_vgg_model(self):
    model= nn.Sequential(nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.AdaptiveAvgPool2d(output_size=(2, 2)))
    return model

  def adapt(self, X_train, y_train):
    # INSERT YOUR CODE HERE
    if self.is_cnn:
      embed          = self.cnn(X_train)
      # print("embed.shape: ", embed.shape)
      embed          = embed.view(embed.shape[0],-1)
      
    else: 
      embed          = self.mlp1(X_train)
  
    unique_labels  = y_train.unique()

    # holds the average embedding vector od each class
    avg_embed      = torch.zeros(len(unique_labels),embed.shape[1]).to(device)

    # print("embed.shape: ", embed.shape)
    # print(avg_embed.shape)

    for idx,label in enumerate(unique_labels):
      avg_embed[idx]= embed[(y_train==label)].mean(dim=0)
    # avg_embed=torch.ravel(avg_embed)
    return avg_embed

  def forward(self,X_test,r):
    # print("r.shape: ", r.shape)
    # print("X_test.shape: ", X_test.shape)
    # INSERT YOUR CODE HERE
    # suppose: X_test=(2,64) ; r=(4,32)
    r=torch.ravel(r) #r=128
    if self.is_cnn: 
      X_test_embed = self.cnn(X_test)
      # X_test_embed = X_test 
      X_test_embed = X_test_embed.view(X_test_embed.shape[0],-1)
      r = r.repeat((X_test_embed.shape[0],1)) #r=(2,128)
      input=torch.cat((r,X_test_embed),dim=1) #input= (2,128+64)
      # print("shape of input: ", input.shape)
    else:
      # X_test_embed = self.mlp1(X_test) #TODO
      # r=r.repeat((X_test_embed.shape[0],1)) #r=(2,128)
      r=r.repeat((X_test.shape[0],1)) 
      input=torch.cat((r,X_test),dim=1) 
      #concatenate the avg_embed and the X_test
      input=torch.cat((r,X_test),dim=1) #input= (2,128+64)

    # print("input: ", input.shape)
    p=self.mlp2(input)
    return p


# Redifning accuracy function so that it takes h - dataset context - as input since net requires it.
def accuracy(Net,X_test,y_test,h,verbose=True):
    Net.eval()
    m = X_test.shape[0]
    y_pred = Net(X_test,h)
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_test).float().sum().item()
    if verbose: print(correct,m)
    accuracy = correct/m
    Net.train()
    return accuracy

def print_optimizer_state(optimizer):
  print(optimizer.__class__)
  logging.info("Optimizer: %s"%(optimizer.__class__))
  dict= optimizer.state_dict()
  logging.info("learning rate: %s, betas: %s, weight_decay: %s"%(dict["param_groups"][0]['lr'], dict["param_groups"][0]["betas"], dict["param_groups"][0]["weight_decay"]))
  print("learning rate: %s, betas: %s, weight_decay: %s"%(dict["param_groups"][0]['lr'], dict["param_groups"][0]["betas"], dict["param_groups"][0]["weight_decay"]))



# net = CNP(n_features=3*64*64,dims=[32,64,32],n_classes=ways, n_channels=3, is_cnn= False).to(device)
net = CNP(n_features=3*64*64,dims=[192,32,8], n_classes=ways, n_channels=3, is_cnn= True).to(device)
#dims= [13312,1024,32]
lossfn = torch.nn.NLLLoss()

# summary(net,[(3,64,64),(1024)])

optimizer    = torch.optim.Adam(net.parameters(),lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


# optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
# print(optimizer.__class__)
# print(optimizer.__class__.__name__)#v
print_optimizer_state(optimizer)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=1e-3, verbose=True)
print(scheduler)
# logger_filename= osp.join(logger_filename, "%s_%d_%s.pth" % (config.data, config.seed, config.model))
# logger_filename= osp.join(logger_filename, "no_seed_mlp.log" )
logger_filename= osp.join(logger_filename, "seed_%s_cnn_%s.log"%(seed,net.is_cnn) )

logging.basicConfig(filename= logger_filename,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filemode='w', level = logging.INFO)
print("Logger file is saved at: ",logger_filename)



######
# from functools import partial
# def lr_sched_maker(lr_sched_epoch_gap, lr_sched_mult, epoch):
#   # if epoch % 30 == 0: return 0.1
#   if epoch % lr_sched_epoch_gap == 0:
#     return lr_sched_mult
#   else:
#     return 1.
# lr_sched = partial(lr_sched_maker, lr_sched_epoch_gap =5, lr_sched_mult=0.1)
# sched = torch.optim.lr_scheduler.MultiplicativeLR(net.optimizer, lr_lambda=lr_sched)
#####



epoch=0
start_time=time()
meta_train_narDS = AnalogicalReasoningDataset(json_filename,nar_filename, meta_train_size1, meta_train_size2)

for epoch in range(n_epochs):
  test_loss = 0.0
  test_acc = 0.0

  for task in range(task_count):
    '''
    d_train= (X_train, X_test)
    d_train[0] = X_train 
    d_train[0][0] is a tensor
    d_train[1] and d_train[1][0] is a tensor.
    
    d_train,d_test=meta_train_kloader.get_task()
    '''

    meta_train_NAR = meta_train_narDS.__getitem__(task)
    d_train,d_test=get_task(meta_train_NAR, net.is_cnn)
    
    rp = torch.randperm(d_train[1].shape[0])
    d_train0=d_train[0][rp].to(device)
    d_train1=d_train[1][rp].to(device)

    h= net.adapt(d_train0,d_train1)

    rp1 = torch.randperm(d_test[1].shape[0])
    d_test0=d_test[0][rp1].to(device)
    d_test1=d_test[1][rp1].to(device)
    

    test_preds = net(d_test0,h)
    # print(test_preds)

    # Accumulate losses over tasks - note train and test loss both included
    test_loss += lossfn(test_preds,d_test1)#+lossfn(train_preds,d_train1)
    test_acc += accuracy(net,d_test0,d_test1,h,verbose=False)
    
  mean_loss = test_loss/task_count
  mean_acc = test_acc/task_count

  logging.info('Epoch  % 2d Loss: %2.5e Avg Acc: %2.5f'%(epoch+1,mean_loss,mean_acc))
  print('Epoch  % 2d Loss: %2.5e Avg Acc: %2.5f'%(epoch+1,mean_loss,mean_acc))
  # display.clear_output(wait=True)
  optimizer.zero_grad()
  test_loss.backward()
  optimizer.step()
  scheduler.step(mean_loss)

  # sched.step()

time_taken=time()-start_time 

logging.info("Time taken for %s is: %s", (n_epochs, np.round(time_taken,3)))
print(f"Time taken for {n_epochs} is: ", np.round(time_taken,3))

test_acc = 0.0

#set the size1 and size2 according to the indices for meta_train_narDS whether you want overlapping or non-overlapping
meta_test_narDS = AnalogicalReasoningDataset(json_filename,nar_filename, meta_test_size1, meta_test_size2)

for task in range(task_count):

    meta_test_NAR = meta_test_narDS.__getitem__(task)
    d_train, d_test = get_task(meta_test_NAR, net.is_cnn)

    d_train0 = d_train[0].to(device)
    d_train1 = d_train[1].to(device)
    h = net.adapt(d_train0,d_train1)
    
    d_test0=d_test[0].to(device)
    d_test1=d_test[1].to(device)
    test_preds = net(d_test0,h)
    test_acc += accuracy(net,d_test0,d_test1,h,verbose=False)
    
    # Done with a task
print('Avg Acc: %2.5f'%(test_acc/task_count))
logging.info('Avg Acc: %2.5f'%(test_acc/task_count))