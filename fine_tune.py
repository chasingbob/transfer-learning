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
    """Generate variations to the input image used by the augmentation step

    # Args:
        img: input image used to generate variations of
        label: the associated label

    """
    X_images = [], y_images = []
    X_images.append(img), 
    y_images.append(label)

    tmp_list = []

    # Flip left-right
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
    
    # change image contrast
    tmp_list[:] = []
    for _img in X_images:
        tmp_list.append( (exposure.rescale_intensity(_img, in_range=(rnd.uniform(0.0, 0.5), rnd.uniform(0.5, 1.0))), label) )

    for _x, _y in tmp_list:
        X_images.append(_x)
        y_images.append(_y)

    return X_images, y_images

def list_to_np(images, labels, image_size=128):
    """Convert list to numpy array and process

    # Args:
        images: the list of images to convert
        labels: the associated labels
        image_size: the desired width/height of each image

    """

    assert len(images) == len(labels)

    X = np.zeros((len(images), image_size, image_size, 3), dtype='float64')
    y = np.zeros((len(labels),))

    count = 0
    for img, label in zip(images, labels):
        img = imresize(img, (image_size, image_size, 3))
        img = np.array(img) / 255.
        X[count] = img
        y[count] = label
        count += 1

    return X, y

def fetch_batch(X, y, iteration, batch_size, image_size, use_augmentation=True):
    """Prepare a batch for training

    # Args
        X: list of images
        y: list of labels
        iteration: number of step to be done
        batch_size: how many images to prepare
        image_size: the desired width/height of each image
        use_augmentation: whether to generate variations or not

    """
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

        _X, _y = list_to_np(tmp_X, tmp_y, image_size)

        return _X, _y
    else:
        _X, _y = list_to_np(X[i:j], y[i:j], image_size)

        return _X, _y

def fetch_files(folder_name, label=0):
    """Fetch all image files in specified folder

    # Args:
        folder_name: name of folder
        label: class label associated with images in folder

    """

    path = os.path.join(data_path, folder_name, '*.jpg')
    files = sorted(glob(path))

    X = [], y = []
    for f in files:
        try:
            img = io.imread(f)
            X.append(img)
            y.append(label)
        except:
            continue
    return X, y

def load_data():
    """Load all images and labels

    # Args:

    """
    print('Load images...')
    x1, y1 = fetch_files(folder_name = 'bastian', label=0)
    x2, y2 = fetch_files(folder_name = 'grace', label=1)
    x3, y3 = fetch_files(folder_name = 'bella', label=2)
    x4, y4 = fetch_files(folder_name = 'pablo', label=3)

    X = [], y = []

    for _x, _y in zip(x1, y1):
        X.append(_x)
        y.append(_y)

    for _x, _y in zip(x2, y2):
        X.append(_x)
        y.append(_y)

    for _x, _y in zip(x3, y3):
        X.append(_x)
        y.append(_y)

    for _x, _y in zip(x4, y4):
        X.append(_x)
        y.append(_y)

    return X, y

def conv_maxpool(inputs, num_filters=32, name='conv-maxpool'):
    """TensorFlow helper method to create a conv layer followed by a maxpool layer

    # Args:
        inputs: input tensor
        num_filters: how many convolutional filters to create
        name: TensorFlow name_scope name

    """
    with tf.name_scope(name):
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=num_filters,
            kernel_size=[3, 3],
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
    """Main method that controls the model training

    # Args:
        model_path: where to load base model
        epochs: how many epochs to train for
        batch_size: number of images in training batch
        image_size: widht/height of image
        learning_rate: rate optimzer is learning at
        feedback_step: how often to give feedback to screen and TensorBoard
        use_augmentation: whether to increase training samples by generating variations

    """
    print('Fine tuning...')

    # Fetch all data, and split in train/validation/test sets
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
    
    # Load tensorflow graph
    saver = tf.train.import_meta_graph(model_path)
    # Access the graph
    #for op in tf.get_default_graph().get_operations():
    #    print(op.name)

    # input/output placeholders
    X = tf.get_default_graph().get_tensor_by_name("placeholders/X:0")
    y = tf.get_default_graph().get_tensor_by_name("placeholders/y:0")

    # Where we want to start fine tuning
    convmax4 = tf.get_default_graph().get_tensor_by_name("model/conv-max-4/MaxPool:0")

    # This will freeze all the layers upto convmax4
    convmax_stop = tf.stop_gradient(convmax4)

    print('Create new top layers')
    with tf.name_scope('new-model'):
        convmax5 = conv_maxpool(inputs=convmax_stop, num_filters=246, name='conv-max-5')
        print('conv-max-5: {}'.format(convmax5.shape))

        with tf.name_scope('flat'):
            new_flat = tf.reshape(convmax5, shape=[-1, 256 * 4 * 4])
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
    step = 0
    print('Session open...')
    with tf.Session() as sess:
        init.run()
        for epoch in range(epochs):
            for iteration in range(len(X_train) // batch_size):
                
                X_batch, y_batch = fetch_batch(X_train, y_train, iteration, batch_size, image_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

                step += 1
                if step % feedback_step == 0:
                    train_acc_str = acc_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    val_acc_str = acc_summary.eval(feed_dict={X: X_val, y: y_val})
                    train_file_writer.add_summary(train_acc_str, step)
                    val_file_writer.add_summary(val_acc_str, step)
                    acc_val = accuracy.eval(feed_dict={X: X_val,y: y_val})
                    acc_train = accuracy.eval(feed_dict={X: X_batch,y: y_batch})
                    print('{}-{} Train acc: {} Val acc: {}'.format(epoch, step,acc_train, acc_val))

        # Calc accuracy against test set
        accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print('Test accuracy: {}'.format(accuracy_test))



if __name__ == "__main__":
    fine_tune()






