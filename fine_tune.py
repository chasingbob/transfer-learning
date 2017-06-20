import os
from datetime import datetime
import random as rnd
from glob import glob
import click
import numpy as np
import tensorflow as tf
from skimage import color, io
from scipy.misc import imresize

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

rnd.seed(45)
root_logdir = "tf_logs"
data_path = './data/dogs'

def fetch_batch(X, iteration, batch_size):
    i = iteration * batch_size
    j = iteration * batch_size + batch_size
    return X[i:j]

def fetch_files(dog_name, image_size=128, label=0):
    path = os.path.join(data_path, dog_name, '*.jpg')

    files = sorted(glob(path))

    file_count = len(files)
    _X = np.zeros((file_count, image_size, image_size, 3), dtype='float64')
    _y = np.zeros((file_count,))
    count = 0
    for f in files:
        try:
            img = io.imread(f)
            new_img = imresize(img, (image_size, image_size, 3))
            new_img = np.array(new_img) / 255.
            _X[count] = new_img
            _y[count] = label
            count += 1
        except:
            continue
    return _X, _y

def load_data():
    print('Load and process images...')
    x1, y1 = fetch_files(dog_name = 'bastian', label=0)
    x2, y2 = fetch_files(dog_name = 'grace', label=1)
    x3, y3 = fetch_files(dog_name = 'bella', label=2)
    x4, y4 = fetch_files(dog_name = 'pablo', label=3)

    _X = np.concatenate(  (x1, np.concatenate( (x2,   np.concatenate( (x3, x4), axis=0)   ), axis=0)), axis=0)
    _y = np.concatenate(  (y1, np.concatenate( (y2,   np.concatenate( (y3, y4), axis=0)   ), axis=0)), axis=0) 

    print('_X shape: {}'.format(_X.shape))
    print('_y shape: {}'.format(_y.shape))

    return _X, _y

def conv_maxpool(inputs, num_filters=32, name='conv-maxpool'):
    with tf.name_scope(name):
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=num_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        return pool

@click.command()
@click.option('--model_path', default='', help='path to base model')
@click.option('--epochs', default=3, help='number of epochs to train model')
@click.option('--batch_size', default=28, help='number of images to go into each training batch')
@click.option('--learning_rate', default=1e-3, help='optimizer learning rate')
@click.option('--feedback_step', default=10, help='write to tensorboard every n-th step')
def fine_tune(model_path, epochs, batch_size, learning_rate, feedback_step):
    print('Fine tuning...')

    X_data, y_data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=44)

    tf.reset_default_graph()
    
    saver = tf.train.import_meta_graph(model_path)
    # Access the graph
    #for op in tf.get_default_graph().get_operations():
    #    print(op.name)

    # input/output placeholders
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    y = tf.get_default_graph().get_tensor_by_name("y:0")

    # get 5th hidden layer
    print('Get conv-max-4...')
    convmax5 = tf.get_default_graph().get_tensor_by_name("model/conv-max-5/MaxPool:0")
    #convmax4 = tf.get_default_graph().get_tensor_by_name("model/conv-max-4/MaxPool:0")
    #print(convmax5.shape)
    # This will freeze all the layers upto convmax5
    convmax_stop = tf.stop_gradient(convmax5)

    print('Create new model')
    with tf.name_scope('new-model'):
        #convmax5 = conv_maxpool(inputs=convmax_stop, num_filters=64, name='conv-max-5')
        #print('conv-max-5: {}'.format(convmax5.shape))

        convmax6 = conv_maxpool(inputs=convmax_stop, num_filters=64, name='conv-max-6')
        print('conv-max-6: {}'.format(convmax6.shape))
    
        with tf.name_scope('flat'):
            new_pool6_flat = tf.reshape(convmax6, shape=[-1, 64 * 2 * 2])

        with tf.name_scope('fc-1'):
            new_dense = tf.layers.dense(inputs=new_pool6_flat, units=1024, activation=tf.nn.relu)
        with tf.name_scope('drop-out-1'):
            new_dropout = tf.layers.dropout(inputs=new_dense, rate=0.5)

        # Logits Layer
        with tf.name_scope('logits-1'):
            new_logits = tf.layers.dense(inputs=new_dropout, units=4)

    print('logits: {}'.format(new_logits.shape))
    print('labels: {}'.format(y.shape))
    with tf.name_scope("new_loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("new_eval"):
        correct = tf.nn.in_top_k(new_logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    with tf.name_scope("new_train"):
        optimizer = tf.train.AdamOptimizer()  
        training_op = optimizer.minimize(loss)

    with tf.name_scope('summary'):
        # This is a bit of a hack to get TensorBoard to display graphs on same chart
        acc_summary = tf.summary.scalar('acc', accuracy)
        val_file_writer = tf.summary.FileWriter('tf_logs/val', tf.get_default_graph())
        train_file_writer = tf.summary.FileWriter('tf_logs/train', tf.get_default_graph())

    init = tf.global_variables_initializer()
    #new_saver = tf.train.Saver()
    step = 0
    print('Session open...')
    with tf.Session() as sess:
        init.run()

        for epoch in range(epochs):
            for iteration in range(len(X_train) // batch_size):
                X_batch = fetch_batch(X_train, iteration, batch_size) 
                y_batch = fetch_batch(y_train, iteration, batch_size) 

                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

                step += 1
                if step % feedback_step == 0:
                    train_acc_str = acc_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    val_acc_str = acc_summary.eval(feed_dict={X: X_test, y: y_test})
                    train_file_writer.add_summary(train_acc_str, step)
                    val_file_writer.add_summary(val_acc_str, step)
                    accuracy_val = accuracy.eval(feed_dict={X: X_test,y: y_test})
                    print('{}-{} Test accuracy: {}'.format(epoch, step,accuracy_val))


def test_predict():
# visualize predictions
    fig=plt.figure()
    fig.set_figheight(18)
    fig.set_figwidth(18)
    
    start = rnd.randint(0, 25)
    for num,img_data in enumerate(X_val[start:start+25]):
        label = np.zeros((1,1))
        label[0] = Y_test[num + start]
        
        _tmp = np.zeros((1, 128, 128, 3), dtype='float32')
        _tmp[0] = img_data
    
        predict = correct.eval(feed_dict={X:_tmp, y:label[0]})
        print('Predict: {} Actual: {}'.format(predict, label[0]))
    
        _sub = fig.add_subplot(5,5,num+1)
    
        str_label = ''
        if predict:
            if label[0] == 0:
                str_label = 'cat'
            else:
                str_label = 'dog'
        else:
            if label[0] == 0:
                str_label = 'dog*'
            else:
                str_label = 'cat*'
        
    
        _sub.imshow(img_data)
        plt.title(str_label, fontsize=18)
        _sub.axes.get_xaxis().set_visible(False)
        _sub.axes.get_yaxis().set_visible(False)
    plt.show()



if __name__ == "__main__":
    fine_tune()






