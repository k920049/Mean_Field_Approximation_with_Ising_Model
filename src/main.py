from __future__ import division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import os

n_epoch = 32
batch_size = 64

def Main():
    mnist = input_data.read_data_sets("../resource")

    with tf.name_scope("placeholder"):
        X = tf.placeholder(dtype=tf.float32, shape=(28, 28))
        
    with tf.name_scope("misc"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()

        for epoch in range(n_epoch):
            for iteration in range(mnist.train.examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                for i in range(batch_size):



if __name__ == "__main__":
    Main()
