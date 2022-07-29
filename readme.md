############   DATASET (Neural Analogical Reasoning)###############

Formulating a cognitive visual problem as meta learning on the Neural Analogical Reasoning Dataset. Available at: https://www.kaggle.com/datasets/gmshroff/few-shot-nar

The dataset has 10k image sets. Each image set has 6 input images and four corresponding options for the output image out of which only one is correct. The input and output image of a single image set is related by the same transformation. The task is to predict the output from among those four options so that the test input - output pair are related by the same transformation. This is a few-shot meta learning problem. 



##################### CNP ################### CNP ###################

HOW TO RUN IT: python CNP/cnp.py

It has two neural networks, 1st: embedding neural network(NN), 2nd: classification neural network(NN)
>Each task has 24 training samples. (24, 3*64*64)
>After embedding NN (24,m)
>Class average embedding (since this is a binary class)(2,m) => flatten this (2m,)
>X_test= (4,3*64*64) (if you plan to pass it through embedding NN then dimension is (4,m) but it gives worse performance, the loss seems to get stuck)
>concatenate X_test and class average embedding (4,2m+3*64*64)
> pass this through classification NN to get predictions


Th frist one is used to get class average embedding and second one is used to predict label for the test image. self.adapt outputs r of dimesion =(no of classes,).

- Things you can tweak : Initially by using my own CNN giving very weird results(results were either getting stuck at 0.25/0.75, maybe stuck at local minima or they would randomly oscillate between 0.25 an d0,75 ). I experiemnted with ADAM and SGD optimizer and ReduceLROnPlateau(optimizer, patience=5, threshold=1e-3, verbose=True) scheduler. It did not give much improvement. 

net = CNP(n_features=3*64*64,dims=[13312,2048,32],n_classes=ways, n_channels=3, is_cnn= True): you can use either the MLP or CNN to extract the embeddings by setting is_cnn. The MLP is the one Sir had given in assignmnet 3 of meta learning course. 

-Next, I tried using pre-trained VGG11  and 16. Since we wanted a less complex model, I went for VGG11. I excluded the classifier module from it, replaced the avgpool with nn.AdaptiveAvgPool2d((1,1)) (so that no of parameters for classification NN will be less) and froze the initial 14 or so layers and finetuned the rest(wanted a relatively complex model than what I hade coded  but also wanted no of parameters less). The aim was to reduce the number of parameters in the model since we have nearly 500 tasks to train (500*24 no of examples). There was overfitting happening- we can try out dropout, regularisation. layers.append(nn.Dropout(0.7)).

In CNP forward, while predicting label for X_test, we can either concatenate it raw or pass it through the previous CNN/MLP to get features and then concatenate. The latter helps in reducing dimension but it did not give good results. why?

task_count=600 : increasing the number of tasks increases the no of training examples(each task gives 24 inputs) will reduce overfitting 
n_epochs=55

############ MAN ############ MAN ############ MAN ####################

HOW TO RUN IT: python MAN/man.py

############ MAML ############ MAML ############ MAML ####################

HOW TO RUN IT: python MAML/maml.py

