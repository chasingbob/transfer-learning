'''CNN model for training on cats vs. dogs dataset to be extendible for finetuning

'''

import os
from glob import glob
import random
import numpy as np
import tensorflow as tf
from skimage import color, io
from scipy.misc import imresize

from base_model import Model

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

    file_count = 25000
    image_holders = np.zeros((file_count, image_size, image_size, 3), dtype='float64')
    label_holders = np.zeros(file_count)
    count = 0
    for filename in cat_files[:12500]:
        try:
            img = io.imread(filename)
            new_img = imresize(img, (image_size, image_size, 3))
            new_img = np.array(new_img) / 255.
            image_holders[count] = new_img
            label_holders[count] = 0
            count += 1
        except:
            continue

    for filename in dog_files[:12500]:
        try:
            img = io.imread(filename)
            new_img = imresize(img, (image_size, image_size, 3))
            new_img = np.array(new_img) / 255.
            image_holders[count] = np.array(new_img)
            label_holders[count] = 1
            count += 1
        except:
            continue
    return image_holders, label_holders

images, labels = load_data()


model = Model()

with tf.Session() as sess:
    model.train(images, labels, sess=sess, num_epochs=30, learning_rate=0.001)
#    model.load('./model-1570-0.84.ckpt', sess=sess)
    for i in range(50):
        r = random.randint(0, 25000)
        result = model.predict(images[r], labels[r])
        print('Label: {} result: {} {}'.format(int(labels[r]), result, int(labels[r]) == result[0]))

    input('Press ENTER to exit')

