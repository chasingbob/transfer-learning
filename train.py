"""Train base model

Custom train a base model on a large data set

"""

import os
from datetime import datetime
import random as rnd
from glob import glob
import numpy as np
import tensorflow as tf
from skimage import color, io
from scipy.misc import imresize
import click

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
import tf_extensions as tfe

rnd.seed(45)

def fetch_batch(data, iteration, batch_size):
    """Fetch data batch for next iteration

    # Args:
        X: data set
        iteration: training step
        batch_size: number of samples to return
    """
    i = iteration * batch_size
    j = iteration * batch_size + batch_size
    return data[i:j]

def load_data(image_size=128):
    """Load all the image files from disk

    # Args:
        image_size: size of image (width = height)

    """
    data_path = './data/train'

    cat_files_path = os.path.join(data_path, 'cat.*.jpg')
    dog_files_path = os.path.join(data_path, 'dog.*.jpg')

    cat_files = sorted(glob(cat_files_path))
    dog_files = sorted(glob(dog_files_path))

    file_count = len(cat_files) + len(dog_files)

    file_count = 2500
    images = np.zeros((file_count, image_size, image_size, 3), dtype='float64')
    labels = np.zeros(file_count)
    count = 0
    for filename in cat_files[:1250]:
        try:
            img = io.imread(filename)
            new_img = imresize(img, (image_size, image_size, 3))
            new_img = np.array(new_img) / 255.
            images[count] = new_img
            labels[count] = 0
            count += 1
        except:
            continue

    for filename in dog_files[:1250]:
        try:
            img = io.imread(filename)
            new_img = imresize(img, (image_size, image_size, 3))
            new_img = np.array(new_img) / 255.
            images[count] = np.array(new_img)
            labels[count] = 1
            count += 1
        except:
            continue
    return images, labels

def test_predict_visual(test_images, test_labels, correct_op, X, y, image_size=128):
    """Visualise predictions

    # Args:
        test_images: test images
        test_labels: labels
        correct_op: tensorflow op, to test if correct
        X: tf input tensor
        y: tf input tensor
        image_size: size of image in pixels (width=height)

    """

    fig = plt.figure()
    fig.set_figheight(18)
    fig.set_figwidth(18)

    start = rnd.randint(0, 25)
    for num, img_data in enumerate(test_images[start:start+25]):
        label = np.zeros((1, 1))
        label[0] = test_labels[num + start]

        _tmp = np.zeros((1, image_size, image_size, 3), dtype='float32')
        _tmp[0] = img_data

        predict = correct_op.eval(feed_dict={X:_tmp, y:label[0]})
        print('Predict: {} Actual: {}'.format(predict, label[0]))

        _sub = fig.add_subplot(5, 5, num+1)

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


@click.command()
@click.option('--epochs', default=3, help='number of epochs to train model')
@click.option('--batch_size', default=16, help='number of images to go into each training batch')
@click.option('--image_size', default=128, help='image size in pixels')
def train(epochs, batch_size, image_size):
    """Main method handling everything around training the model

    # Args:
        epochs: Number of epochs to train model
        batch_size: Number of images in each training batch send for training
        image_size: Image size in pixels (width = height)

    """
    print('Training started...')

    all_X, ally = load_data(image_size=image_size)

    X_train, X_val, Y_train, Y_val = train_test_split(all_X, ally, test_size=0.10, random_state=41)
    X_val, X_test, y_val, y_test = train_test_split(X_val, Y_val, test_size=0.5, random_state=99)
    print('Train/Val/Test split:')
    print('X_train: {} {}'.format(X_train.shape[0], X_train.shape))
    print('X_val: {} {}'.format(X_val.shape[0], X_val.shape))
    print('X_test: {} {}'.format(X_test.shape[0], X_test.shape))

    with tf.name_scope('placeholders'):
        with tf.name_scope('X'):
            X = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name="X")
        with tf.name_scope('y'):
            y = tf.placeholder(tf.int32, shape=[None], name="y")

    with tf.name_scope('model'):
        # layer 1
        conv1 = tfe.conv(inputs=X, num_filters=64, name='conv-1')
        pool1 = tfe.maxpool(inputs=conv1, name='maxpool-1')

        # layer 2 
        conv2 = tfe.conv(inputs=pool1, num_filters=128, name='conv-2')
        pool2 = tfe.maxpool(inputs=conv2, name='maxpool-2')

        # layer 3 
        conv3 = tfe.conv(inputs=pool2, num_filters=256, name='conv-3')
        pool3 = tfe.maxpool(inputs=conv3, name='maxpool-3')

        # layer 4 
        conv4 = tfe.conv(inputs=pool3, num_filters=512, name='conv-4')
        pool4 = tfe.maxpool(inputs=conv4, name='maxpool-4')

        print('last pool shape: {}'.format(pool4.shape))

        with tf.name_scope('flat'):
            pool_flat = tf.reshape(pool4, shape=[-1, 512 * 8 * 8])

        with tf.name_scope('fc-1'):
            fc1 = tf.layers.dense(inputs=pool_flat, units=2048, activation=tf.nn.relu)
        
        with tf.name_scope('drop-out-1'):
            dropout = tf.layers.dropout(inputs=fc1, rate=0.5)

        # Logits Layer
        with tf.name_scope('logits-1'):
            logits = tf.layers.dense(inputs=dropout, units=2)

    with tf.name_scope('ops-1'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('summary'):
        # accuracy
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

        current_acc = tf.Variable(0.0, name="current_acc")
        acc_summary = tf.summary.scalar('acc', current_acc)
        val_file_writer = tf.summary.FileWriter('tf_logs/val', tf.get_default_graph())
        train_file_writer = tf.summary.FileWriter('tf_logs/train', tf.get_default_graph())

        write_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    step = 0
    prev_best = 0
    with tf.Session() as sess:
        init.run()
        for epoch in range(epochs):
            for i in range(len(X_train) // batch_size):
                X_train_batch = fetch_batch(X_train, i, batch_size)
                Y_train_batch = fetch_batch(Y_train, i, batch_size)
                
                sess.run(training_op, feed_dict={X: X_train_batch, y: Y_train_batch})
                
                step += 1
                val_accs = []
                if step % 50 == 0:
                    # TensorBoard feedback step
                    val_accs[:] = []

                    for j in range(len(X_val) // batch_size):
                        X_val_batch = fetch_batch(X_val, j, batch_size)
                        y_val_batch = fetch_batch(y_val, j, batch_size)

                        val_acc = sess.run(accuracy, feed_dict={X:X_val_batch, y: y_val_batch})
                        val_accs.append(val_acc)
                    
                    temp_acc = sum(val_accs)/len(val_accs)
                    _summary = sess.run(write_op, {current_acc: temp_acc})
                    val_file_writer.add_summary(_summary, step)
                    val_file_writer.flush()

                    train_acc = sess.run(accuracy, feed_dict={X:X_train_batch, y: Y_train_batch})
                    _summary = sess.run(write_op, {current_acc: train_acc})
                    train_file_writer.add_summary(_summary, step)
                    train_file_writer.flush()

                    print('{}-{} Train acc: {} Val acc: {}'.format(epoch, step, train_acc, temp_acc))

                    if temp_acc > prev_best:
                        print('... save')
                        prev_best = temp_acc
                        save_path = saver.save(sess, "./model-{}-{:2.2f}.ckpt".format(epoch, temp_acc))

        test_accs = []
        for i in range(len(X_test) // batch_size):
            X_test_batch = fetch_batch(X_test, j, batch_size)
            y_test_batch = fetch_batch(y_test, j, batch_size)

            test_acc = sess.run(accuracy, feed_dict={X:X_test_batch, y: y_test_batch})
            test_accs.append(test_acc)
    
        print('Test acc: {}'.format(sum(test_accs)/len(test_accs)))

        test_predict_visual(X_test, y_test, correct, X, y, image_size)

    val_file_writer.close()

    

if __name__ == "__main__":
