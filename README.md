# transfer-learning
exploring ways to learn with the least amount of labelled data by taking advantage of transfer-learning techniques

I've set myself out to classify images of my dogs using deep learning using the least amount of training photos. 
Fine tuning an existing model is the first part of a 3x part series exploring different ways of using Transfer Learning to accomplish training with little data.

![Bastian](images/bastian.jpg | width=150) ![Bella](images/bella.jpg | width=150) ![Grace](images/grace.jpg | width=150) ![Pablo](images/pablo.jpg | width=150)

Who wouldn't want to train on them? :-)


## Part 1: Fine-tune existing custom model

The idea is to fine-tune a model trained on a large publicly available data set. I chose the Kaggle cats vs. dogs data set with 25000 images (12500 cats, 12500 dogs) hoping that my model will learn enough about dogs to be able to fine tune the model on a small set of my dog images.


![Architecture](images/transfer-learning-custom-model.png)

