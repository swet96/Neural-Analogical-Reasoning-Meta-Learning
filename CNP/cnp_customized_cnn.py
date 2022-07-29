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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Accelerator found: ", device)

class CNP(nn.Module):
    def __init__(self, n_features, dims, n_classes, n_channels, is_cnn= False): #data=(X_test,(X_train,y_train))
      super(CNP,self).__init__()

      self.n_features   = n_features
      self.n_classes    = n_classes
      self.is_cnn       = is_cnn
      if is_cnn:
        print("Using CNN to get average class embeddings!")
        self.cnn =models.CNN(n_channels = n_channels)
        dims= dims+[n_classes]
        self.mlp2 = models.MLP(dims=dims)
      else:
        print("Using MLP to get average class embeddings!")
        dimL1             = [n_features]+dims
        self.mlp1         = models.MLP(dims=dimL1,task='embedding')
        dimL2             = [n_features+n_classes*dims[-1]]+dims+[n_classes] #TODO
        # dimL2             = [dims[-1]+n_classes*dims[-1]]+dims+[n_classes]
        self.mlp2         = models.MLP(dims=dimL2, task= 'classification')

      self.optimizer    = torch.optim.Adam(self.parameters(),lr=1e-4)
      print(self.optimizer.__class__)
      # self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)


    def adapt(self, X_train, y_train):
      # INSERT YOUR CODE HERE
      if self.is_cnn:
        embed          = self.cnn(X_train)
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
      return avg_embed

    def forward(self,X_test,r):
      # INSERT YOUR CODE HERE
      # suppose: X_test=(2,64) ; r=(4,32)
      r=torch.ravel(r) #r=128
      if self.is_cnn: 
        X_test_embed = self.cnn(X_test)
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

        
      # print(input.shape)
      p=self.mlp2(input)
      return p


def set_seed(seed):
  random.seed(0) #0
  np.random.seed(0) #0
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

seed = 17
set_seed(seed)
logger_root = f"logger_files"

ways=2
task_count=200

meta_train_size1 = 0
meta_train_size2 =task_count

meta_test_size1 = task_count
meta_test_size2 = 2*task_count


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


# net = CNP(n_features=3*64*64,dims=[32,64,32],n_classes=ways, n_channels=3, is_cnn= False).to(device)
net = CNP(n_features=3*64*64,dims=[1536,1024,128,32],n_classes=ways, n_channels=3, is_cnn= True).to(device)
lossfn = torch.nn.NLLLoss()

# logger_filename= osp.join(logger_root, "%s_%d_%s.pth" % (config.data, config.seed, config.model))
# logger_filename= osp.join(logger_root, "no_seed_mlp.log" )
logger_filename= osp.join(logger_root, "seed_%s_cnn_%s.log"%(seed,net.is_cnn) )

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
n_epochs=40
start_time=time()
meta_train_narDS = AnalogicalReasoningDataset("./problems_10k.json","./nar_classification_dataset_images_10k/images_large", meta_train_size1, meta_train_size2)
for epoch in range(n_epochs):
  test_loss = 0.0
  test_acc = 0.0
  # Sample and train on a task
  
  for task in range(task_count):
    '''
    d_train= (X_train, X_test)
    d_train[0] = X_train and d_train[0][0] is a tensor
    d_train[1] and d_train[1][0] is a tensor.
    
    d_train,d_test=meta_train_kloader.get_task()
    '''

    meta_train_NAR = meta_train_narDS.__getitem__(task)
    d_train,d_test=get_task(meta_train_NAR, net.is_cnn)
    
    ########################################
    # rp = torch.randperm(d_train[1].shape[0])
    # d_train0=d_train[0][rp]
    # d_train1=d_train[1][rp]
    # x_tr = d_train0
    # d_tr = x_tr 
    # h= net.adapt(d_tr,d_train1)

    # rp1 = torch.randperm(d_test[1].shape[0])
    # d_test0=d_test[0][rp1]
    # d_test1=d_test[1][rp1]
    # x_ts = d_test0
    # y_ts_sh = torch.zeros(x_ts.shape[0],ways)
    # d_ts = x_ts 

    # test_preds = net(d_ts,h)


    rp = torch.randperm(d_train[1].shape[0])
    d_train0=d_train[0][rp].to(device)
    d_train1=d_train[1][rp].to(device)
    h= net.adapt(d_train0,d_train1)

    rp1 = torch.randperm(d_test[1].shape[0])
    d_test0=d_test[0][rp1].to(device)
    d_test1=d_test[1][rp1].to(device)
    

    test_preds = net(d_test0,h)


    ############################################
    # Accumulate losses over tasks - note train and test loss both included
    test_loss += lossfn(test_preds,d_test1)#+lossfn(train_preds,d_train1)
    test_acc += accuracy(net,d_test0,d_test1,h,verbose=False)
    
    
  #Update the network weights
  logging.info('Epoch  % 2d Loss: %2.5e Avg Acc: %2.5f'%(epoch+1,test_loss/task_count,test_acc/task_count))
  print('Epoch  % 2d Loss: %2.5e Avg Acc: %2.5f'%(epoch+1,test_loss/task_count,test_acc/task_count))
  # display.clear_output(wait=True)
  net.optimizer.zero_grad()
  test_loss.backward()
  net.optimizer.step()

  # sched.step()

time_taken=time()-start_time 

print(f"Time taken for {n_epochs} is: ", np.round(time_taken,3))
logging.info(f"Time taken for {n_epochs} is: ", np.round(time_taken,3))

test_acc = 0.0
adapt_steps = 1
#set the size1 and size2 according to the indices for meta_train_narDS whether you want overlapping or non-overlapping
meta_test_narDS = AnalogicalReasoningDataset("./problems_10k.json","./nar_classification_dataset_images_10k/images_large", meta_test_size1, meta_test_size2)

# Sample and train on a task
for task in range(task_count):
    # d_train,d_test=meta_test_kloader.get_task()

    meta_test_NAR = meta_test_narDS.__getitem__(task)
    d_train, d_test = get_task(meta_test_NAR, net.is_cnn)
    ########################################
    # x_tr = d_train[0]
    # y_tr_sh = torch.cat((torch.zeros(1,ways),torch.eye(ways)[d_train[1][1:]]))
    # d_tr = x_tr #torch.cat((x_tr,y_tr_sh),1)
    # h=net.adapt(d_tr,d_train[1])
    # x_ts = d_test[0]
    # y_ts_sh = torch.zeros(x_ts.shape[0],ways)
    # d_ts = x_ts #torch.cat((x_ts,y_ts_sh),1)
    # test_preds = net(d_ts,h)
    # test_acc += accuracy(net,d_ts,d_test[1],h,verbose=False)
    ##########################################

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