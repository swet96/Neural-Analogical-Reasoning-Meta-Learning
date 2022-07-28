import sys
sys.path.append("/home/sweta/sweta_NAR_paper")

import learn2learn as l2l
import torch.optim as optim
import models
import torch
# torch.cuda.empty_cache()
import numpy as np
from time import time
from dataset import AnalogicalReasoningDataset
import random 
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import sys
import logging
import os.path as osp


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device= torch.device("cpu")
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
logger_filename = f"./logger_files_maml" 
json_filename= f"./problems_10k.json"
nar_filename= f"./nar_classification_dataset_images_10k/images_large"

ways=2
task_count=500
n_epochs=30

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


class CNN(nn.Module):
  def __init__(self):
      super(CNN, self).__init__()
      self.out_size = 128*4*4
      self.cnn= self.get_vgg_model()
      

  def get_vgg_model(self):
      model = torchvision.models.vgg11(pretrained=True)
      model.features=nn.Sequential(list(model.children())[0][:6])
      model.avgpool= nn.AdaptiveAvgPool2d((4,4))
      model.classifier = nn.Sequential(nn.Linear(self.out_size,32),nn.Dropout(0.5), nn.ReLU(), nn.Linear(32, 2),nn.Dropout(0.5),nn.LogSoftmax(dim=1))
      # for c in list(model.children())[0][:14]:
      #     for p in c.parameters():
      #         p.requires_grad = False
      # print("Model used to get features: ", model)
      return model

  def forward(self,x):
      # res = self.cnn(x)
      # return res.view(x.shape[0],-1)
      return self.cnn(x)

# net = models.MLP(dims=[3*64*64,32,32,ways]).to(device)
net= CNN().to(device)

if str(net.__class__).split(".")[1][:3]=="MLP":
  is_cnn= False
else:
  is_cnn= True

logger_filename= osp.join(logger_filename, "seed_%s_cnn_%s.log"%(seed,is_cnn) )

logging.basicConfig(filename= logger_filename,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filemode='w', level = logging.INFO)
print("Logger file is saved at: ",logger_filename)


def print_optimizer_state(optimizer):
  print(optimizer.__class__)
  logging.info("Optimizer: %s"%(optimizer.__class__))
  dict= optimizer.state_dict()
  logging.info("learning rate: %s, betas: %s, weight_decay: %s"%(dict["param_groups"][0]['lr'], dict["param_groups"][0]["betas"], dict["param_groups"][0]["weight_decay"]))
  print("learning rate: %s, betas: %s, weight_decay: %s"%(dict["param_groups"][0]['lr'], dict["param_groups"][0]["betas"], dict["param_groups"][0]["weight_decay"]))

print("CNN model used: %s "%(net))
logging.info("CNN model used: %s "%(net))


maml = l2l.algorithms.MAML(net, lr=5e-3)
logging.info("MAML model used: %s "%(maml))
print("MAML model used: %s "%(maml))

# optimizer = optim.Adam(maml.parameters(),lr=5e-3)
optimizer = optim.Adam(maml.parameters(), lr=0.01, betas=(0.7, 0.88), eps=1e-08, weight_decay=00)
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.8, dampening=0.8, weight_decay=0)
# optimizer= torch.optim.RMSprop(net.parameters(), lr=0.0001, alpha=0.8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor= 0.2, min_lr=1e-4, threshold=1e-3, verbose=True)
lossfn = torch.nn.NLLLoss()
print_optimizer_state(optimizer)
print(scheduler)


epoch=0
fas = 6
meta_train_narDS = AnalogicalReasoningDataset(json_filename,nar_filename,meta_train_size1,meta_train_size2)

start_time=time()

while epoch<n_epochs:
  adapt_loss = 0.0
  test_acc = 0.0
  # Sample and train on a task
  for task in range(task_count):
    meta_train_NAR = meta_train_narDS.__getitem__(task)
    d_train, d_test = get_task(meta_train_NAR, is_cnn)

    
    d_train0=d_train[0].to(device)    
    # print(d_train0.dtype)
    # print("/n")              
    d_train1=d_train[1].to(device)
    d_test0=d_test[0].to(device)  
    d_test1=d_test[1].to(device)

    learner = maml.clone()
    for fas_step in range(fas):
        train_preds = learner(d_train0)
        # print("train_preds.shape: ", train_preds.shape)
        # print(train_preds)
        train_loss = lossfn(train_preds,d_train1)
        learner.adapt(train_loss)

    test_preds = learner(d_test0)
    adapt_loss += lossfn(test_preds,d_test1)
    learner.eval()
    test_acc += models.accuracy(learner,d_test0,d_test1,verbose=False)
    learner.train()
      
  # Update main network
  print('Epoch  % 2d Loss: %2.5e Avg Acc: %2.5f'%(epoch+1,adapt_loss/task_count,test_acc/task_count))
  logging.info('Epoch  % 2d Loss: %2.5e Avg Acc: %2.5f'%(epoch+1,adapt_loss/task_count,test_acc/task_count))

  # display.clear_output(wait=True)
  optimizer.zero_grad()
  total_loss = adapt_loss
  total_loss.backward()
  optimizer.step()
  scheduler.step(total_loss/task_count)
  epoch+=1
time_taken=time()-start_time 

print(f"Time taken for {n_epochs} is: ", np.round(time_taken,3))
logging.info("Time taken for %s is: %s "% (n_epochs, np.round(time_taken,3)))

test_acc = 0.0
adapt_steps = 6
meta_test_narDS = AnalogicalReasoningDataset(json_filename,nar_filename, meta_test_size1, meta_test_size2)

maml.eval()
# Sample and train on a task
for task in range(task_count):
    meta_test_NAR = meta_test_narDS.__getitem__(task)
    d_train, d_test = get_task(meta_test_NAR, is_cnn)
    
    d_train0=d_train[0].to(device)                  
    d_train1=d_train[1].to(device)
    d_test0=d_test[0].to(device)  
    d_test1=d_test[1].to(device)

    learner = maml.clone()
    learner.eval()
    for adapt_step in range(adapt_steps):
        train_preds = learner(d_train0)
        train_loss = lossfn(train_preds,d_train1)
        learner.adapt(train_loss)
    test_preds = learner(d_test0)
    test_acc += models.accuracy(learner,d_test0,d_test1,verbose=False)
    # Done with a task
#learner.train()
print('Avg Acc: %2.5f'%(test_acc/task_count))
logging.info('Avg Acc: %2.5f'%(test_acc/task_count))