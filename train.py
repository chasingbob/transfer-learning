import os
from datetime import datetime
import random as rnd
from glob import glob
import numpy as np
import tensorflow as tf
from skimage import color, io
from scipy.misc import imresize

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

rnd.seed(45)

def fetch_batch(X, iteration, batch_size):
    i = iteration * batch_size
    j = iteration * batch_size + batch_size
    return X[i:j]

data_path = './data/train'

cat_files_path = os.path.join(data_path, 'cat.*.jpg')
dog_files_path = os.path.join(data_path, 'dog.*.jpg')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))

file_count = len(cat_files) + len(dog_files)
print(file_count)

image_size = 128

file_count = 20000
allX = np.zeros((file_count, image_size, image_size, 3), dtype='float64')
ally = np.zeros(file_count)
count = 0
for f in cat_files[:10000]:
    try:
        img = io.imread(f)
        new_img = imresize(img, (image_size, image_size, 3))
        new_img = np.array(new_img) / 255.
        allX[count] = new_img
        ally[count] = 0
        count += 1
    except:
        continue

for f in dog_files[:10000]:
    try:
        img = io.imread(f)
        new_img = imresize(img, (image_size, image_size, 3))
        new_img = np.array(new_img) / 255.
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue
        
file_count = count
        
# test-train split   
X_train, X_val, Y_train, Y_val = train_test_split(allX, ally, test_size=0.05, random_state=43)
print('X: {} {}'.format(X_train.shape[0], X_train.shape))
print('y: {} {}'.format(Y_train.shape[0], Y_train.shape))  

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_path = 'tf_logs'
logdir = '{}/run-{}/'.format(root_path, now)

X = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name="X")
y = tf.placeholder(tf.int32, shape=[None], name="y")

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

with tf.name_scope('model'):
    convmax1 = conv_maxpool(inputs=X, num_filters=32, name='conv-max-1')
    convmax2 = conv_maxpool(inputs=convmax1, num_filters=64, name='conv-max-2')
    convmax3 = conv_maxpool(inputs=convmax2, num_filters=64, name='conv-max-3')
    convmax4 = conv_maxpool(inputs=convmax3, num_filters=64, name='conv-max-4')
    convmax5 = conv_maxpool(inputs=convmax4, num_filters=32, name='conv-max-5')
    convmax6 = conv_maxpool(inputs=convmax5, num_filters=64, name='conv-max-6')

    print(convmax6.shape)

    with tf.name_scope('flat'):
        pool6_flat = tf.reshape(convmax6, shape=[-1, 64 * 2 * 2])

    with tf.name_scope('fc-1'):
        dense = tf.layers.dense(inputs=pool6_flat, units=1024, activation=tf.nn.relu)
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

with tf.name_scope('feedback'):
    # accuracy
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    # TensorBoard
    val_acc_summary = tf.summary.scalar('val_acc', accuracy)
    train_acc_summary = tf.summary.scalar('train_acc', accuracy)



val_file_writer = tf.summary.FileWriter('tf_logs/log2/', tf.get_default_graph())
train_file_writer = tf.summary.FileWriter('tf_logs/log2/', tf.get_default_graph())

# Init
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 150
batch_size = 256

avg_acc = []
train_accs = []
test_accs = []


def train():
    init.run()
    step = 0
    prev_best = 0
    for epoch in range(n_epochs):
        n_batches = len(X_train) // batch_size
        for i in range(len(X_train) // batch_size):
            X_train_batch = fetch_batch(X_train, i, batch_size)
            Y_train_batch = fetch_batch(Y_train, i, batch_size)
            
            sess.run(training_op, feed_dict={X: X_train_batch, y: Y_train_batch})
            
            step += 1
            if step % 100 == 0:
                # TensorBoard feedback step
                val_str = val_acc_summary.eval(feed_dict={X: X_val, y: Y_val})
                train_str = train_acc_summary.eval(feed_dict={X: X_train_batch, y: Y_train_batch})
                
                val_file_writer.add_summary(val_str, step)
                train_file_writer.add_summary(train_str, step)
                
                #acc_train = accuracy.eval(feed_dict={X: X_batch, y: Y_batch})
                #acc_test = accuracy.eval(feed_dict={X: X_val, y: Y_test})
                
        rand = rnd.randint(0, n_batches-1)
        X_batch = fetch_batch(X_train, rand, batch_size)
        Y_batch = fetch_batch(Y_train, rand, batch_size)
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: Y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_val, y: Y_val})
        
        train_accs.append(acc_train)
        test_accs.append(acc_test)
        print(epoch, 'Train acc: {}-{} Val acc: {}'.format(rand, acc_train, acc_test))

        if acc_test > prev_best:
            print('... save')
            prev_best = acc_test
            save_path = saver.save(sess, "./model-{}-{:2.2f}.ckpt".format(epoch, acc_test))


def load_model():
    saver.restore(sess, './model-9-0.74.ckpt')

        
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

with tf.Session() as sess:
    train()
#    load_model()
#    test_predict()

    
val_file_writer.close()