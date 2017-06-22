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

rnd.seed(45)

def fetch_batch(X, iteration, batch_size):
    i = iteration * batch_size
    j = iteration * batch_size + batch_size
    return X[i:j]

def load_data():
    data_path = './data/train'

    cat_files_path = os.path.join(data_path, 'cat.*.jpg')
    dog_files_path = os.path.join(data_path, 'dog.*.jpg')

    cat_files = sorted(glob(cat_files_path))
    dog_files = sorted(glob(dog_files_path))

    file_count = len(cat_files) + len(dog_files)
    print(file_count)

    image_size = 128

    file_count = 4000
    allX = np.zeros((file_count, image_size, image_size, 3), dtype='float64')
    ally = np.zeros(file_count)
    count = 0
    for f in cat_files[:2000]:
        try:
            img = io.imread(f)
            new_img = imresize(img, (image_size, image_size, 3))
            new_img = np.array(new_img) / 255.
            allX[count] = new_img
            ally[count] = 0
            count += 1
        except:
            continue

    for f in dog_files[:2000]:
        try:
            img = io.imread(f)
                new_img = imresize(img, (image_size, image_size, 3))
            new_img = np.array(new_img) / 255.
            allX[count] = np.array(new_img)
            ally[count] = 1
            count += 1
        except:
            continue
    return allX, ally
            
    file_count = count
            
    # test-train split   
    

    return X_tr

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_path = 'tf_logs'
logdir = '{}/run-{}/'.format(root_path, now)



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

#avg_acc = []
#train_accs = []
#test_accs = []

@click.command()
@click.option('--epochs', default=3, help='number of epochs to train model')
@click.option('--batch_size', default=16, help='number of images to go into each training batch')
def train(epochs, batch_size):
    print('Start Training...')

    allX, ally = load_data()

    X_train, X_val, Y_train, Y_val = train_test_split(allX, ally, test_size=0.5, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_val, Y_val, test_size=0.5, random_state=97)
    print('Train/Val/Test split:')
    print('X_train: {} {}'.format(X_train.shape[0], X_train.shape))
    print('X_val: {} {}'.format(X_val.shape[0], X_val.shape))
    print('X_test: {} {}'.format(X_test.shape[0], X_test.shape))

    with tf.name_scope('placeholders'):
        X = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name="X")
        y = tf.placeholder(tf.int32, shape=[None], name="y")

    with tf.name_scope('model'):
        convmax1 = conv_maxpool(inputs=X, num_filters=32, name='conv-max-1')
        convmax2 = conv_maxpool(inputs=convmax1, num_filters=64, name='conv-max-2')
        convmax3 = conv_maxpool(inputs=convmax2, num_filters=128, name='conv-max-3')
        convmax4 = conv_maxpool(inputs=convmax3, num_filters=128, name='conv-max-4')
        #convmax5 = conv_maxpool(inputs=convmax4, num_filters=128, name='conv-max-5')
        #convmax6 = conv_maxpool(inputs=convmax5, num_filters=128, name='conv-max-6')

        print('Convmax 4 shape: {}'.format(convmax4.shape))

        with tf.name_scope('flat'):
            pool_flat = tf.reshape(convmax4, shape=[-1, 128 * 8 * 8])

        with tf.name_scope('fc-1'):
            dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
        with tf.name_scope('drop-out-1'):
            dropout = tf.layers.dropout(inputs=dense, rate=0.5)

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
        # This is a bit of a hack to get TensorBoard to display graphs on same chart
        #acc_summary = tf.summary.scalar('acc', accuracy)

        current_acc = tf.Variable(0.0, name="current_acc")
        acc_summary = tf.summary.scalar('acc', current_acc)
        val_file_writer = tf.summary.FileWriter('tf_logs/val', tf.get_default_graph())
        train_file_writer = tf.summary.FileWriter('tf_logs/train', tf.get_default_graph())

        write_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    init.run()
    step = 0
    prev_best = 0
    for epoch in range(epochs):
        for i in range(len(X_train) // batch_size):
            X_train_batch = fetch_batch(X_train, i, batch_size)
            Y_train_batch = fetch_batch(Y_train, i, batch_size)
            
            sess.run(training_op, feed_dict={X: X_train_batch, y: Y_train_batch})
            
            step += 1
            val_accs = []
            if step % 10 == 0:
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
        print('Epoch: {}'.format(epoch))
    print('No more epochs!!')

    val_file_writer.close()


def load_model():
    saver.restore(sess, './model-9-0.74.ckpt')

def test_accuracy():
    print('*** Test acc ***')
    test_accs = []
    for i in range(len(X_test) // batch_size):
        X_test_batch = fetch_batch(X_test, j, batch_size)
        y_test_batch = fetch_batch(y_test, j, batch_size)

        test_acc = sess.run(accuracy, feed_dict={X:X_test_batch, y: y_test_batch})
        test_accs.append(test_acc)
    
    print('Model accuracy against test set: {}'.format(sum(test_accs)/len(test_accs)))
               
     
def test_predict_visual():
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
    print('__main__')
    with tf.Session() as sess:
        train()
    #    load_model()
        #print('Test acc now...')
        #test_accuracy()
    #    test_predict_visual(sess)

    print('__main__')
    

    