from __future__ import division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

from model.Ising import Ising

import os

n_epoch = 32
batch_size = 64

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def Main():
    mnist = input_data.read_data_sets("../resource")

    with tf.name_scope("placeholder"):
        X = tf.placeholder(dtype=tf.float32, shape=(28, 28))

    with tf.name_scope("misc"):
        init = tf.global_variables_initializer()
#         saver = tf.train.Saver()
    img = mpimg.imread(fname="../resource/image.jpg", format='jpg')
    img = Image.fromarray(img).convert('L')
    img.show()
    img = np.array(object=img)
    print(img.shape)

    ising = Ising(img)
    ising.set_disp_flag(True)
    ising.run_iteration()
'''
    with tf.Session() as sess:
        init.run()

        X_batch, y_batch = mnist.train.next_batch(1)
        X_batch = np.reshape(a=X_batch, newshape=(28, 28))
        X_batch = X_batch * 255


        for epoch in range(n_epoch):
            for iteration in range(mnist.train.examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                for i in range(batch_size):

'''


if __name__ == "__main__":
    Main()
