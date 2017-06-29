"""Collection of tensorflow extention methods

Keep all the tensorflow extention methods in one place

"""

import tensorflow as tf

def conv(inputs, num_filters=32, name='conv'):
    """Convolutional layer

    # Args:
        inputs: input layer
        num_filters: number of kernels/filters to use
        name: tf name_scope name

    """
    with tf.name_scope(name):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=num_filters,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

def maxpool(inputs, name='maxpool'):
    """Max pool layer

    # Args:
        inputs: input layer
        name: tf name_scope name
   """
    with tf.name_scope(name):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        