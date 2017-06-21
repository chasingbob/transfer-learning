# transfer-learning
> Exploring ways to learn with the least amount of labelled data by taking advantage of transfer-learning techniques.

I've set myself the challenge to classify images of my dogs using the least amount of training photos. There 

### Meet the dogs

![Bastian](images/bastian.jpg) ![Bella](images/bella.jpg) ![Grace](images/grace.jpg) ![Pablo](images/pablo.jpg)

Who wouldn't want to train on them? :-)


## Option 1: Fine-tune existing custom model

The idea is to fine-tune a model trained on a large publicly available data set. I chose the Kaggle cats vs. dogs data set with 25000 images (12500 cats, 12500 dogs) hoping that my model will learn enough about the features that are unique to dogs to be able to fine tune the model on a small set of my dog images.


![Architecture](images/transfer-learning-custom-model.png)


## Using the code

Dependencies:

* Python 3.5
* numpy
* scikit-learn
* matplotlib
* tensorflow (1.1)

Create the following folder structure:

transfer-learning
(the source code lives here)

|

|----data

|    (the train/test data lives here)

|

|

|----tf_logs

     (the TensorBoard logs lives here)

Download the cats vs. dogs dataset from [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) and extract into a folder called train under the data folder.

Train custom model:
* Start Tensorboard

        In a terminal:
        ```tensorboard --logdir tf_logs/
        ```

* Train the model:

        In a terminal:
        ```python3 train.py 
        ```




## Release History

* 0.0.1
    * Work in progress



## Get in touch

Dries Cronje - [@dries139](twitter.com/dries139) - dries.cronje@outlook.com




