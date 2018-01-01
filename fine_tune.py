'''Fine-tune existing model

Fine tune an existing model on a small data set by freezing bottom layers and
training on the top layers by using a small learning rate.

'''

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
    '''Generate variations to the input image used by the augmentation step

    # Args:
        img: input image used to generate variations of
        label: the associated label

    '''

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
        for _ in range(4):
            from_x = int(rnd.uniform(0.0, 0.25) * width)
            from_y = int(rnd.uniform(0.0, 0.25) * height)
            to_x = int((0.75 + rnd.uniform(0.0, 0.25)) * width)
            to_y = int((0.75 + rnd.uniform(0.0, 0.25)) * height)

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
    '''Convert list to numpy array and process

    # Args:
        images: the list of images to convert
        labels: the associated labels
        image_size: the desired width/height of each image

    '''

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
    '''Prepare a batch for training

    # Args
        X: list of images
        y: list of labels
        iteration: number of step to be done
        batch_size: how many images to prepare
        image_size: the desired width/height of each image
        use_augmentation: whether to generate variations or not

    '''
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


def fetch_images(folder_name, label=0):
    '''Fetch all image files in specified folder

    # Args:
        folder_name: name of folder
        label: class label associated with images in folder

    '''

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
    '''Load all images and labels

    '''

    print('Load images...')

    images1, labels1 = fetch_images(folder_name='bastian', label=0)
    print('Found {} Bastian images'.format(len(images1)))
    images2, labels2 = fetch_images(folder_name='grace', label=1)
    print('Found {} Grace images'.format(len(images2)))
    images3, labels3 = fetch_images(folder_name='bella', label=2)
    print('Found {} Bella images'.format(len(images3)))
    images4, labels4 = fetch_images(folder_name='pablo', label=3)
    print('Found {} Pablo images'.format(len(images4)))

    images = []
    labels = []

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
@click.option('--epochs', default=30, help='number of epochs to train model')
@click.option('--batch_size', default=28, help='number of images to go into each training batch')
@click.option('--image_size', default=128, help='fixed size of image')
@click.option('--learning_rate', default=1e-3, help='optimizer learning rate')
@click.option('--feedback_step', default=20, help='write to tensorboard every n-th step')
@click.option('--use_augmentation', is_flag=True, help='increase image pool by using augmentation')
@click.option('--option', default='train', help='training or inference')
def fine_tune(option, model_path, epochs, batch_size, image_size, learning_rate, feedback_step, use_augmentation):
    print('Augmentation: {}'.format(use_augmentation))
    if option == 'inference':
        visualise_test_predictions(model_path)

    elif option == 'train':
        train(model_path, epochs, batch_size, image_size, feedback_step, use_augmentation)



def train(model_path, epochs, batch_size, image_size, feedback_step, use_augmentation):
    '''Main method that controls the model training

    # Args:
        model_path: where to load base model
        epochs: how many epochs to train for
        batch_size: number of images in training batch
        image_size: widht/height of image
        learning_rate: rate optimzer is learning at
        feedback_step: how often to give feedback to screen and TensorBoard
        use_augmentation: whether to increase training samples by generating variations

    '''
    print('Fine tuning...')

    # Fetch all data, and split in train/validation/test sets
    X_data, y_data = load_data()

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, random_state=3)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.55, random_state=55)

    X_val, y_val = list_to_np(X_val, y_val, image_size)
    X_test, y_test = list_to_np(X_test, y_test, image_size)


    tf.reset_default_graph()
    
    # Load tensorflow graph
    saver = tf.train.import_meta_graph(model_path)
    # Access the graph
#    for op in tf.get_default_graph().get_operations():
#        print(op.name)


    # input/output placeholders
    X = tf.get_default_graph().get_tensor_by_name("placeholders/X:0")
    y = tf.get_default_graph().get_tensor_by_name("placeholders/y:0")

    # Where we want to start fine tuning
    pool3 = tf.get_default_graph().get_tensor_by_name("model/maxpool-3/MaxPool:0")

    # This will freeze all the layers upto convmax4
    maxpool_stop = tf.stop_gradient(pool3)

    print('Create new top layers')
    with tf.name_scope('new-model'):
        conv4 = tfe.conv(inputs=maxpool_stop, num_filters=512, name='new-conv-4')
        pool4 = tfe.maxpool(inputs=conv4, name='new-maxpool-4')
        print('pool4: {}'.format(pool4.shape))

        with tf.name_scope('flat'):
            new_flat = tf.reshape(pool4, shape=[-1, 512 * 8 * 8])
        with tf.name_scope('fc-1'):
            fc1 = tf.layers.dense(inputs=new_flat, units=2048, activation=tf.nn.relu)
        with tf.name_scope('drop-out-1'):
            new_dropout = tf.layers.dropout(inputs=fc1, rate=0.5)

        # Logits Layer
        with tf.name_scope('logits-1'):
            new_logits = tf.layers.dense(inputs=new_dropout, units=4)

    with tf.name_scope("new_loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("new_eval"):
        correct = tf.nn.in_top_k(new_logits, y, 1, name='correct')
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
    saver = tf.train.Saver()
    step = 0
    print('Session open...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        init.run()
        best_acc = 0.0
        for epoch in range(epochs):
            for iteration in range(len(X_train) // batch_size):

                X_batch, y_batch = fetch_batch(X_train, y_train, iteration, batch_size, image_size, use_augmentation=use_augmentation)
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

                    if acc_val > best_acc:
                        best_acc = acc_val
                        saver.save(sess, "models/finetune-model-{}-{:2.2f}.ckpt".format(epoch, acc_val))

        # Calc accuracy against test set
        accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print('Test accuracy: {}'.format(accuracy_test))

def visualise_test_predictions(file_name):

    tf.reset_default_graph()
    
    # Load tensorflow graph
    saver = tf.train.import_meta_graph(file_name)

     # input/output placeholders
    X = tf.get_default_graph().get_tensor_by_name("placeholders/X/X:0")
    y = tf.get_default_graph().get_tensor_by_name("placeholders/y/y:0")

    for op in tf.get_default_graph().get_operations():
        print(op.name)

    correct_op = tf.get_default_graph().get_tensor_by_name("new_eval")

    fig = plt.figure()
    fig.set_figheight(18)
    fig.set_figwidth(18)

    # Load model
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    X_data, y_data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=5)
    X_test, y_test = list_to_np(X_test, y_test, 128)

    X_test = X_test[:25]
    y_test = y_test[:25]

    # Init session
    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, file_name)

        for num, img_data in enumerate(X_test):
            label = np.zeros((1, 1))
            label[0] = y_test[num]

            _tmp = np.zeros((1, 128, 128, 3), dtype='float32')
            _tmp[0] = img_data

            predict = correct_op.eval(feed_dict={X:_tmp, y:label[0]})
            print('Predict: {} Actual: {}'.format(predict, label[0]))

            _sub = fig.add_subplot(5, 5, num+1)

            str_label = ''
            if predict:
                if label[0] == 0:
                    str_label = 'Bastian'
                if label[0] == 1:
                    str_label = 'Grace'
                if label[0] == 2:
                    str_label = 'Bella'
                else:
                    str_label = 'Pablo'
            else:
                if label[0] == 0:
                    str_label = 'Bastian**'
                if label[0] == 1:
                    str_label = 'Grace**'
                if label[0] == 2:
                    str_label = 'Bella**'
                else:
                    str_label = 'Pablo**'


        _sub.imshow(img_data)
        plt.title(str_label, fontsize=18)
        _sub.axes.get_xaxis().set_visible(False)
        _sub.axes.get_yaxis().set_visible(False)
    plt.show()


    # Run predictions

    # Visualise predictions




if __name__ == "__main__":
    fine_tune()





