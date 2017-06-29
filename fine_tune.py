"""Fine-tune existing model

Fine tune an existing model on a small data set by freezing bottom layers and
training on the top layers by using a small learning rate.

"""

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
import tf_extensions as tfe

rnd.seed(47)
root_logdir = "tf_logs"
data_path = './data/dogs'

def get_img_variations(img, label):
    """Generate variations to the input image used by the augmentation step

    # Args:
        img: input image used to generate variations of
        label: the associated label

    """
    X_images = []; y_images = []
    X_images.append(img), 
    y_images.append(label)
    tmp_list = []

    # Flip left-right
    for _img in X_images:
        tmp_list.append((np.fliplr(_img), label))

    for _x, _y in tmp_list:
        X_images.append(_x)
        y_images.append(_y)
    tmp_list[:] = []

    # random crops
    for _img in X_images:
        width, height, _ = _img.shape
        for _ in range(2):
            from_x = int(rnd.uniform(0.0, 0.25) * width)
            from_y = int(rnd.uniform(0.0, 0.25) * height)
            to_x = int((0.5 + rnd.uniform(0.0, 0.25)) * width)
            to_y = int((0.5 + rnd.uniform(0.0, 0.25)) * height)

            tmp_list.append((_img[from_y:to_y, from_x:to_x], label))

    for _x, _y in tmp_list:
        X_images.append(_x)
        y_images.append(_y)

    # change image contrast
    tmp_list[:] = []
    for _img in X_images:
        tmp_list.append((exposure.rescale_intensity(
            _img,
            in_range=(rnd.uniform(0.1, 0.5), rnd.uniform(0.5, 0.9))), label))

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

    X = np.zeros((len(images), image_size, image_size, 3), dtype='float32')
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
    if use_augmentation:
        images = []
        labels = []

        for _x, _y in zip(X[i:j], y[i:j]):
            xs, ys = get_img_variations(_x, _y)
            for _images, _labels in zip(xs, ys):
                images.append(_images)
                labels.append(_labels)

        return list_to_np(images, labels, image_size)

    else:
        return list_to_np(X[i:j], y[i:j], image_size)


def fetch_files(folder_name, label=0):
    """Fetch all image files in specified folder

    # Args:
        folder_name: name of folder
        label: class label associated with images in folder

    """

    path = os.path.join(data_path, folder_name, '*.jpg')
    files = sorted(glob(path))

    images = []; labels = []
    for f in files:
        try:
            img = io.imread(f)
            images.append(img)
            labels.append(label)
        except:
            continue
    return images, labels

def load_data():
    """Load all images and labels

    # Args:

    """

    print('Load images...')

    images1, labels1 = fetch_files(folder_name='bastian', label=0)
    images2, labels2 = fetch_files(folder_name='grace', label=1)
    images3, labels3 = fetch_files(folder_name='bella', label=2)
    images4, labels4 = fetch_files(folder_name='pablo', label=3)

    images = []; labels = []

    for _x, _y in zip(images1, labels1):
        images.append(_x)
        labels.append(_y)

    for _x, _y in zip(images2, labels2):
        images.append(_x)
        labels.append(_y)

    for _x, _y in zip(images3, labels3):
        images.append(_x)
        labels.append(_y)

    for _x, _y in zip(images4, labels4):
        images.append(_x)
        labels.append(_y)

    return images, labels

@click.command()
@click.option('--model_path', default='', help='path to base model')
@click.option('--epochs', default=10, help='number of epochs to train model')
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

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.25, random_state=23)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.57, random_state=56)

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
    X = tf.get_default_graph().get_tensor_by_name("placeholders/X/X:0")
    y = tf.get_default_graph().get_tensor_by_name("placeholders/y/y:0")

    # Where we want to start fine tuning
    pool4 = tf.get_default_graph().get_tensor_by_name("model/maxpool-4/MaxPool:0")

    # This will freeze all the layers upto convmax4
    maxpool_stop = tf.stop_gradient(pool4)

    print('Create new top layers')
    with tf.name_scope('new-model'):
        conv5 = tfe.conv(inputs=maxpool_stop, num_filters=512, name='conv-5')
        pool5 = tfe.maxpool(inputs=conv5, name='maxpool-5')
        print('pool5: {}'.format(pool5.shape))

        with tf.name_scope('flat'):
            new_flat = tf.reshape(pool5, shape=[-1, 512 * 4 * 4])
        with tf.name_scope('fc-1'):
            fc1 = tf.layers.dense(inputs=new_flat, units=1000, activation=tf.nn.relu)
        with tf.name_scope('drop-out-1'):
            new_dropout = tf.layers.dropout(inputs=fc1, rate=0.5)

        # Logits Layer
        with tf.name_scope('logits-1'):
            new_logits = tf.layers.dense(inputs=new_dropout, units=4)

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
                    acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
                    acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                    print('{}-{} Train acc: {} Val acc: {}'.format(epoch, step,acc_train, acc_val))

        # Calc accuracy against test set
        accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print('Test accuracy: {}'.format(accuracy_test))



if __name__ == "__main__":
    #fine_tune()






