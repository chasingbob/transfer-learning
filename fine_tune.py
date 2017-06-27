import os
from datetime import datetime
import random as rnd
from glob import glob
import click
import numpy as np
import tensorflow as tf
from skimage import color, io, exposure
from scipy.misc import imresize

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

rnd.seed(47)
root_logdir = "tf_logs"
data_path = './data/dogs'

def get_img_variations(img, label):
    X_images = []
    X_images.append(img)
    y_images = []
    y_images.append(label)

    tmp_list = []
    for _img in X_images:
        tmp_list.append( (np.fliplr(_img), label) )

    for _x, _y in tmp_list:
        X_images.append(_x)
        y_images.append(_y)
    
    tmp_list[:] = []

    # random crops
    for _img in X_images:
        w, h, _ = _img.shape
        from_x = int(rnd.uniform(0.0, 0.25) * w)
        from_y = int(rnd.uniform(0.0, 0.25) * h)
        to_x = int((0.75 + rnd.uniform(0.0, 0.25)) * w)
        to_y = int((0.75 + rnd.uniform(0.0, 0.25)) * h)

        tmp_list.append( (_img[from_y:to_y,from_x:to_x], label) )

    for _x, _y in tmp_list:
        X_images.append(_x)
        y_images.append(_y)
    
    tmp_list[:] = []

    for _img in X_images:
        tmp_list.append( (exposure.rescale_intensity(_img, in_range=(rnd.uniform(0.0, 0.5), rnd.uniform(0.5, 1.0))), label) )

    for _x, _y in tmp_list:
        X_images.append(_x)
        y_images.append(_y)

    return X_images, y_images

def list_to_np(images, labels, image_size=128):
    assert len(images) == len(labels)

    _X = np.zeros((len(images), image_size, image_size, 3), dtype='float64')
    _y = np.zeros((len(labels),))

    count = 0
    for img, label in zip(images, labels):
        img = imresize(img, (image_size, image_size, 3))
        img = np.array(img) / 255.
        _X[count] = img
        _y[count] = label
        count += 1

    return _X, _y

def fetch_batch(X, y, iteration, batch_size, image_size, use_augmentation=True):
    i = iteration * batch_size
    j = iteration * batch_size + batch_size
    if use_augmentation==True:
        tmp_X = []
        tmp_y = []

        for _x, _y in zip(X[i:j], y[i:j]):
            xs, ys = get_img_variations(_x, _y)
            for _tmp_X, _tmp_y in zip(xs, ys):
                tmp_X.append(_tmp_X)
                tmp_y.append(_tmp_y)

        #print(len(tmp_X))
        
        _X, _y = list_to_np(tmp_X, tmp_y, image_size)

        return _X, _y
    else:
        _X, _y = list_to_np(X[i:j], y[i:j], image_size)

        return _X, _y

def fetch_files2(dog_name, image_size=128, label=0):
    path = os.path.join(data_path, dog_name, '*.jpg')
    files = sorted(glob(path))

    _X = []
    _y = []
    count = 0
    for f in files:
        try:
            img = io.imread(f)
            _X.append(img)
            _y.append(label)
            count += 1
        except:
            continue
    return _X, _y

def load_data():
    print('Load and process images...')
    x1, y1 = fetch_files2(dog_name = 'bastian', label=0)
    x2, y2 = fetch_files2(dog_name = 'grace', label=1)
    x3, y3 = fetch_files2(dog_name = 'bella', label=2)
    x4, y4 = fetch_files2(dog_name = 'pablo', label=3)

    _X = []
    _y = []

    for x, y in zip(x1, y1):
        _X.append(x)
        _y.append(y)

    for x, y in zip(x2, y2):
        _X.append(x)
        _y.append(y)

    for x, y in zip(x3, y3):
        _X.append(x)
        _y.append(y)

    for x, y in zip(x4, y4):
        _X.append(x)
        _y.append(y)

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
@click.option('--image_size', default=128, help='fixed size of image')
@click.option('--learning_rate', default=1e-3, help='optimizer learning rate')
@click.option('--feedback_step', default=50, help='write to tensorboard every n-th step')
@click.option('--use_augmentation', default=True, help='increase image pool by using augmentation')
def fine_tune(model_path, epochs, batch_size, image_size, learning_rate, feedback_step, use_augmentation):
    print('Fine tuning...')

    X_data, y_data = load_data()

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.27, random_state=26)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.55, random_state=59)

    X_val, y_val = list_to_np(X_val, y_val, image_size)
    X_test, y_test = list_to_np(X_test, y_test, image_size)

    print('X_test: {}'.format(X_test.shape))
    print('y_test: {}'.format(y_test.shape))
    print('X_val: {}'.format(X_val.shape))
    print('y_val: {}'.format(y_val.shape))

    tf.reset_default_graph()
    
    saver = tf.train.import_meta_graph(model_path)
    # Access the graph
    #for op in tf.get_default_graph().get_operations():
    #    print(op.name)

    # input/output placeholders
    X = tf.get_default_graph().get_tensor_by_name("placeholders/X:0")
    y = tf.get_default_graph().get_tensor_by_name("placeholders/y:0")

    # get 4th hidden layer
    print('Get conv-max-4...')
    convmax4 = tf.get_default_graph().get_tensor_by_name("model/conv-max-4/MaxPool:0")

    # This will freeze all the layers upto convmax5
    convmax_stop = tf.stop_gradient(convmax4)
    print('convmax_stop: {}'.format(convmax_stop.shape))

    print('Create new top layers')
    with tf.name_scope('new-model'):
        convmax5 = conv_maxpool(inputs=convmax_stop, num_filters=128, name='conv-max-5')
        print('conv-max-5: {}'.format(convmax5.shape))

        with tf.name_scope('flat'):
            new_flat = tf.reshape(convmax5, shape=[-1, 128 * 4 * 4])
        with tf.name_scope('fc-1'):
            fc1 = tf.layers.dense(inputs=new_flat, units=1024, activation=tf.nn.relu)
        with tf.name_scope('fc-2'):
            fc2 = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu)
        with tf.name_scope('drop-out-1'):
            new_dropout = tf.layers.dropout(inputs=fc2, rate=0.5)

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
        optimizer = tf.train.AdamOptimizer(learning_rate)  
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
                
                X_batch, y_batch = fetch_batch(X_train, y_train, iteration, batch_size, image_size)
                #print('X_batch: {} y_batch: {}'.format(X_batch.shape, y_batch.shape))
                
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

                step += 1
                if step % feedback_step == 0:
                    train_acc_str = acc_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    val_acc_str = acc_summary.eval(feed_dict={X: X_val, y: y_val})
                    train_file_writer.add_summary(train_acc_str, step)
                    val_file_writer.add_summary(val_acc_str, step)
                    accuracy_val = accuracy.eval(feed_dict={X: X_val,y: y_val})
                    accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
                    print('{}-{} Val acc: {} Test acc: {}'.format(epoch, step,accuracy_val, accuracy_test))

        # Calc accuracy against test set
        accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print('Test accuracy: {}'.format(accuracy_test))


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






