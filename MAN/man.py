
import sys
sys.path.append("/home/sweta/sweta_NAR_paper")

from extras import MatchingNetwork
import torch.optim as optim
import models
import torch
import torch.nn as nn
from dataset import AnalogicalReasoningDataset
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import logging
import os.path as osp
import torchvision  #ADDED
from torchsummary import summary

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Accelerator found: ", device)

def set_seed(seed):
  random.seed(0) #0
  np.random.seed(0) #0
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

seed = 17 
print("seed used is: ", seed)
set_seed(seed)
logger_filename = f"./logger_files_man" 
json_filename= f"./problems_10k.json"
nar_filename= f"./nar_classification_dataset_images_10k/images_large"

ways=2
task_count=500
n_epochs=60 #60
print("(Tasks are non-overlapping) Number of tasks used for meta train and meta test: %s, Number of epochs: %s"%(task_count, n_epochs))

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

def print_optimizer_state(optimizer):
  print(optimizer.__class__)
  logging.info("Optimizer: %s"%(optimizer.__class__))
  dict= optimizer.state_dict()
  logging.info("learning rate: %s, betas: %s, weight_decay: %s"%(dict["param_groups"][0]['lr'], dict["param_groups"][0]["betas"], dict["param_groups"][0]["weight_decay"]))
  print("learning rate: %s, betas: %s, weight_decay: %s"%(dict["param_groups"][0]['lr'], dict["param_groups"][0]["betas"], dict["param_groups"][0]["weight_decay"]))

net = MatchingNetwork(num_channels=3, is_cnn= True).to(device) #is_cnn is used for the get_task which gives images of the form (N,C,H,W) or (N,_) depending if MLP or CNN is used
print("Model used to get features: ", net)
logging.info("Model used to get features: ", net)

# summary(net,[(3,64,64),(1024)])


optimizer    = torch.optim.Adam(net.parameters(),lr=1e-1)
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
print_optimizer_state(optimizer)

# optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
# print(optimizer.__class__)
# print(optimizer.__class__.__name__)#v

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=1e-3, verbose=True)
print(scheduler)

# logger_filename= osp.join(logger_filename, "%s_%d_%s.pth" % (config.data, config.seed, config.model))
# logger_filename= osp.join(logger_filename, "no_seed_mlp.log" )
logger_filename= osp.join(logger_filename, "seed_%s_cnn_%s.log"%(seed,net.is_cnn) )

logging.basicConfig(filename= logger_filename,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filemode='w', level = logging.INFO)
print("Logger file is saved at: ",logger_filename)


epoch=0
start_time=time()
meta_train_narDS = AnalogicalReasoningDataset("./problems_10k.json","./nar_classification_dataset_images_10k/images_large", meta_train_size1, meta_train_size2)

for epoch in range(n_epochs):
  test_loss = 0.0
  test_acc = 0.0

  for task in range(task_count):

    meta_train_NAR = meta_train_narDS.__getitem__(task)
    d_train, d_test = get_task(meta_train_NAR, net.is_cnn)

    rp = torch.randperm(d_train[1].shape[0]) #so that the samples and labels are shuffled in the similar way
    d_train0=d_train[0][rp].to(device)                  #samples are randomly shuffled
    d_train1=d_train[1][rp].to(device)

    rp1 = torch.randperm(d_test[1].shape[0]) 
    d_test0=d_test[0][rp1].to(device)  
    d_test1=d_test[1][rp1].to(device)

    accuracy, loss = net(d_train0, d_train1, d_test0, d_test1)
    # print(test_preds)
    
    # Accumulate losses over tasks - note train and test loss both included
    test_loss += loss
    test_acc += accuracy
     
  mean_loss = test_loss/task_count
  mean_acc = test_acc/task_count


  logging.info('Epoch  % 2d Loss: %2.5e Avg Acc: %2.5f'%(epoch+1,mean_loss,mean_acc))
  print('Epoch  % 2d Loss: %2.5e Avg Acc: %2.5f'%(epoch+1,mean_loss,mean_acc))
  # display.clear_output(wait=True)
  optimizer.zero_grad()
  test_loss.backward()
  optimizer.step()
  scheduler.step(mean_loss)

time_taken=time()-start_time 

logging.info("Time taken for %s is: %s" % (n_epochs, np.round(time_taken,3)))
print(f"Time taken for {n_epochs} is: ", np.round(time_taken,3))

test_acc = 0.0

#set the size1 and size2 according to the indices for meta_train_narDS whether you want overlapping or non-overlapping
meta_test_narDS = AnalogicalReasoningDataset(json_filename,nar_filename, meta_test_size1, meta_test_size2)


for task in range(task_count):
    meta_test_NAR = meta_test_narDS.__getitem__(task)
    d_train, d_test = get_task(meta_test_NAR, net.is_cnn)
    
    d_train0 = d_train[0].to(device)
    d_train1 = d_train[1].to(device)

    d_test0=d_test[0].to(device)
    d_test1=d_test[1].to(device)
    accuracy, loss = net(d_train0, d_train1, d_test0, d_test1)
    
    test_acc += accuracy

    
print('Avg Acc: %2.5f'%(test_acc/task_count))
logging.info('Avg Acc: %2.5f'%(test_acc/task_count))