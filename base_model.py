'''Base model architecture for Cats vs. dogs classification

'''

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tf_extensions as tfe
import utils


class Model:
    '''Base model class


    '''

    def __init__(self, image_size=128):
        with tf.name_scope('placeholders'):
            self.X = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name="X")
            self.y = tf.placeholder(tf.int32, shape=[None], name="y")

        with tf.name_scope('model'):
            # layer 1
            self.conv1 = tfe.conv(inputs=self.X, num_filters=64, name='conv-1')
            self.pool1 = tfe.maxpool(inputs=self.conv1, name='maxpool-1')

            # layer 2
            self.conv2 = tfe.conv(inputs=self.pool1, num_filters=128, name='conv-2')
            self.pool2 = tfe.maxpool(inputs=self.conv2, name='maxpool-2')

            # layer 3
            self.conv3 = tfe.conv(inputs=self.pool2, num_filters=256, name='conv-3')
            self.pool3 = tfe.maxpool(inputs=self.conv3, name='maxpool-3')

            # layer 4
            self.conv4 = tfe.conv(inputs=self.pool3, num_filters=512, name='conv-4')
            self.pool4 = tfe.maxpool(inputs=self.conv4, name='maxpool-4')

            # print('last pool shape: {}'.format(self.pool4.shape))

            with tf.name_scope('flat'):
                self.pool_flat = tf.reshape(self.pool4, shape=[-1, 512 * 8 * 8])

            with tf.name_scope('fc-1'):
                self.fc1 = tf.layers.dense(inputs=self.pool_flat, units=2048, activation=tf.nn.relu)

            with tf.name_scope('drop-out-1'):
                self.dropout = tf.layers.dropout(inputs=self.fc1, rate=0.5)

            # Logits Layer
            with tf.name_scope('logits-1'):
                self.logits = tf.layers.dense(inputs=self.dropout, units=2)

            with tf.name_scope('training-ops'):
                self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.y)
                self.loss = tf.reduce_mean(self.xentropy)
                optimizer = tf.train.AdamOptimizer()
                self.training_op = optimizer.minimize(self.loss)

            with tf.name_scope('summary'):
                self.correct = tf.nn.in_top_k(self.logits, self.y, 1)
                self.prediction = tf.argmax(self.logits, 1)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32), name='accuracy')

                self.current_acc = tf.Variable(0.0, name="current_acc")
                self.acc_summary = tf.summary.scalar('acc', self.current_acc)
                self.val_file_writer = tf.summary.FileWriter(
                    'tf_logs/val',
                    tf.get_default_graph())
                self.train_file_writer = tf.summary.FileWriter(
                    'tf_logs/train',
                    tf.get_default_graph())

                self.current_loss = tf.Variable(0.0, name="current_loss")
                self.loss_summary = tf.summary.scalar('loss', self.current_loss)
#                self.val_file_writer = tf.summary.FileWriter('tf_logs/val', tf.get_default_graph())
#                self.train_file_writer = tf.summary.FileWriter(
#                    'tf_logs/train',
#                    tf.get_default_graph())

                self.write_op = tf.summary.merge_all()


            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()



    def train(self, images, labels, sess=None, num_epochs=5, batch_size=56, learning_rate=0.001):
        '''Train model on supplied inputs

            # Args:
                images: input images, [-1, 128, 128, 3]
                labels: input labels, [-1, 1]


        '''
        if sess is None:
            raise Exception('Valid tf session should be passed in ')

        X_train, X_val, Y_train, Y_val = train_test_split(
            images,
            labels,
            test_size=0.10,
            random_state=41)
        X_val, X_test, y_val, y_test = train_test_split(
            X_val,
            Y_val,
            test_size=0.5,
            random_state=99)

        self.init.run()
        step = 0
        val_accs = []
        prev_best = 0

        for epoch in range(num_epochs):
            for i in range(len(X_train) // batch_size):
                X_train_batch = utils.fetch_batch(X_train, i, batch_size)
                Y_train_batch = utils.fetch_batch(Y_train, i, batch_size)

                sess.run(self.training_op, feed_dict={self.X: X_train_batch, self.y: Y_train_batch})

                step += 1
                if step % 10 == 0:
                    # feedback step
                    val_accs[:] = []

                    for j in range(len(X_val) // batch_size):
                        X_val_batch = utils.fetch_batch(X_val, j, batch_size)
                        y_val_batch = utils.fetch_batch(y_val, j, batch_size)

                        val_acc = sess.run(self.accuracy, feed_dict={self.X:X_val_batch, self.y: y_val_batch})
                        val_accs.append(val_acc)

                    train_acc = sess.run(self.accuracy, feed_dict={self.X:X_train_batch, self.y: Y_train_batch})
                    temp_acc = sum(val_accs)/len(val_accs)
                    print('{}-{} Train: {} Val: {}'.format(epoch, step, train_acc, temp_acc))

                    _summary = sess.run(self.write_op, {self.current_acc: temp_acc})
                    self.val_file_writer.add_summary(_summary, step)
                    self.val_file_writer.flush()

                    _summary = sess.run(self.write_op, {self.current_acc: train_acc})
                    self.train_file_writer.add_summary(_summary, step)
                    self.train_file_writer.flush()

                    if temp_acc > prev_best:
                        prev_best = temp_acc
                        print('Saved...')
                        self.saver.save(sess, "./model-{}-{:2.2f}.ckpt".format(step, temp_acc))


    def predict(self, image, label):
        '''Predict label from input image

            #Args:
                image: [128, 128, 3] normalized [0,1]
                label: []

        '''

        img = np.zeros((1, 128, 128, 3), dtype='float32')
        img[0] = image
        _label = np.zeros((1, 1))
        _label[0] = label

        predict = self.prediction.eval(feed_dict={self.X:img, self.y:_label[0]})
        return predict


    def load(self, file_name, sess=None):
        '''Load model

            #Args:
                file_name: name of model to load
                sess: active tensorflow session

        '''

        if sess is None:
            raise Exception('Valid tf session should be passed in ')

        self.init.run()

        self.saver.restore(sess, file_name)
