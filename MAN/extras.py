import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



     

class AttentionalClassify(nn.Module):
  def __init__(self):
      super(AttentionalClassify, self).__init__()

  def forward(self, similarities, d_train1):
      """
      Products pdfs over the support set classes for the target set image.
      :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
      :param support_set_y:[batch_size,sequence_length,classes_num]
      :return: Softmax pdf shape[batch_size,classes_num]
      """
      softmax     =nn.Softmax(dim=1) # (4,)
    
      similarities_softmax=softmax(similarities) #shape =(4,24) # log softmax of the similarities returned by cos method
    
      unique_labels =d_train1.unique()
      # print(unique_labels)
      preds         =torch.empty(similarities.shape[0],len(unique_labels))
    
    
      for i,label in enumerate(unique_labels):
        preds[:,i]=torch.log(similarities_softmax[:,d_train1==label].sum(dim=1))
      # print(preds.shape)
      return preds




class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()
    def forward(self,target,ss):
        #traget= (4,m), support set= (24,m)
        '''
        example= D_train and D_test of D_meta_train is 5-shot-5-ways 
        target.shape= (batch,embedding_dim) , here batch=5*5=25, ss_size=5*5=25 
        '''
        # compute cosine similarities between 
        # target (batch,embedding_dim) and support set ss (ss_size,embedding_dim)
        # return (batch,ss_size)
        # INSERT YOUR CODE HERE

        cos_sim=nn.CosineSimilarity(dim=1, eps=1e-8)
        similarities=torch.zeros(target.shape[0],ss.shape[0])
        
        for i in range(target.shape[0]):
        #target[i] = (1,m) and repaeted to have same no of rows as ss =(24,m) , then cos sim is found between target and ss
        # print("target[i].shape: ",target[i].shape)
        # print("ss.shape[0]: ",ss.shape[0])
            similarities[i]=cos_sim(target[i].repeat(ss.shape[0],1),ss)  # each row contains similarity value of one x_test data  with the train_data
        
        # similarities = (4,24)
        # print("shape of similarities: ", similarities.shape)
        return similarities
    

class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_size, batch_size, vector_dim,use_cuda):
        super(BidirectionalLSTM, self).__init__()
        """
        Initial a muti-layer Bidirectional LSTM
        :param layer_size: a list of each layer'size
        :param batch_size: 
        :param vector_dim: 
        """
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = self.init_hidden(self.use_cuda)

    def init_hidden(self,use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda(),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda())
        else:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False))

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        # self.hidden = self.init_hidden(self.use_cuda)
        self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn= self.get_vgg_model()
        self.out_size = 25088

    # def get_vgg_model(self):
    #     model = torchvision.models.vgg11(pretrained=True)
    #     model.classifier = nn.Sequential(nn.Linear(25088,64), nn.ReLU(), nn.Linear(64, 32))
    #     for c in list(model.children())[0][:10]:
    #         for p in c.parameters():
    #             p.requires_grad = False
    #     print("Model used to get features: ", model)
    #     return model

    def get_vgg_model(self):
        model = torchvision.models.vgg11(pretrained=True)
        model.features = nn.Sequential(nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        model.classifier = nn.Sequential(nn.Linear(64,32))
        # for c in list(model.children())[0][:10]:
        #     for p in c.parameters():
        #         p.requires_grad = False
        return model

    def forward(self,x):
        res = self.cnn(x)
        return res.view(x.shape[0],-1)

class MatchingNetwork(nn.Module):
    def __init__(self,num_channels=3, is_cnn= False): #use_cuda=True, fce=True, learning_rate=1e-3,batch_size=28,
        """
        This is our main network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.is_cnn= is_cnn
        # self.batch_size = batch_size
        # self.keep_prob = keep_prob
        self.num_channels = num_channels
        # self.learning_rate = learning_rate
        # self.fce = fce
        # self.num_classes_per_set = num_classes_per_set
        # self.num_samples_per_class = num_samples_per_class
        # self.image_size = image_size
        self.cnn = CNN()
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        # if self.fce:
        #     self.lstm = BidirectionalLSTM(layer_size=[32], batch_size=self.batch_size, vector_dim=self.cnn.out_size, use_cuda=use_cuda)

    def forward(self, d_train0, d_train1, d_test0, d_test1):
        """ d_train0, d_train1, d_test0, d_test1
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:
        :return:
        """
        # produce embeddings for support set images
        
        ss_embedding = self.cnn(d_train0)
        target_embedding = self.cnn(d_test0)
        output= torch.vstack((ss_embedding, target_embedding))

        # []
        # for i in np.arange(support_set_images.size(0)):
        #     gen_encode = self.cnn(support_set_images[:, i, :, :])
        #     encoded_images.append(gen_encode)

        # produce embeddings for target images
        # gen_encode = self.cnn(d_test0)
        # encoded_images.append(gen_encode)
        # output = torch.stack(encoded_images)

        # use fce?
        # if self.fce:
        #     output = self.lstm(output)

        # get similarities between support set embeddings and target
        similarities = self.dn(target=output[24:,:], ss=output[:24,:])
        
        # produce predictions for target probabilities
        preds = self.classify(similarities, d_train1)
        
        # calculate the accuracy
        values, indices = preds.max(1)
        indices = indices.to(device)
        preds = preds.to(device)
        accuracy = torch.mean((indices.squeeze() == d_test1).float())
        crossentropy_loss = F.cross_entropy(preds, d_test1.long())

        return accuracy, crossentropy_loss