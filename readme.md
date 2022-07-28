
##################### DIRECTORY STRUCTURE ##########################################
In "sweta_NAR_paper":

-dataset.py
-models.py
-logger_files_cnp
-logger_files_maml
-logger_files_man
-problems_10k.json
-nar_classification_dataset_images_10k

-CNP
  -cnp.py

-MAML
  -maml.py

-MAN
  -man.py
  -extras.py




##################### CNP ################### CNP ###################
HOW TO RUN IT: python CNP/cnp.py

It has two neural networks, 1st: embedding neural network(NN), 2nd: classification neural network(NN)
>Each task has 24 training samples. (24, 3*64*64)
>After embedding NN (24,m)
>Class average embedding (2,m) => flatten this (2m,)
>X_test= (4,3*64*64) (if you plan to pass it through embedding NN then (4,m) but it gives worse performance, the loss seems to get stuck)
>concatenate X_test and class average embedding (4,2m+3*64*64)
> pass this through classification NN


Th frist one is used to get class average embedding and second one is used to predict label for the test image. self.adapt outputs r (no of classes,)

- Things you can tweak : Initially by using my own CNN giving very weird results(results were either getting stuck at 0.25/0.75, maybe stuck at local minima or they would randomly oscillate between 0.25 an d0,75 ). I experiemnted with ADAM and SGD optimizer and ReduceLROnPlateau(optimizer, patience=5, threshold=1e-3, verbose=True) scheduler. It did not give much improvement. 

net = CNP(n_features=3*64*64,dims=[13312,2048,32],n_classes=ways, n_channels=3, is_cnn= True): you can use either the MLP or CNN to extract the embeddings by setting is_cnn. The MLP is the one Sir had given in assignmnet 3 of meta learning course. 

Next, I tried using pre-trained VGG11  and 16. Since we wanted a less complex model, I went for VGG11. I excluded the classifier module from it, replaced the avgpool with nn.AdaptiveAvgPool2d((1,1)) (so that no of parameters for classification NN will be less) and froze the initial 14 or so layers and finetuned the rest(wanted a relatively complex model than what I hade coded  but also wanted no of parameters less). The aim was to reduce the number of parameters in the model since we have nearly 500 tasks to train (500*24 no of examples). There was overfitting happening- we can try out dropout, regularisation. layers.append(nn.Dropout(0.7)).

In CNP forward, while predicting label for X_test, we can either concatenate it raw or pass it through the previous CNN/MLP to get features and then concatenate. The latter helps in reducing dimension but it did not give good results. why?

task_count=600 : increasing the task nos, hence no of training examples(each task gives 24 inputs) will reduce overfitting 
n_epochs=55





############ MAN ############ MAN ############ MAN ####################
HOW TO RUN IT: python MAN/man.py

get embedding train data = (batch_size,m)= (24,m) m= no of features after passing through the embedding NN
get embedding test data = (4,m)
for each of the 