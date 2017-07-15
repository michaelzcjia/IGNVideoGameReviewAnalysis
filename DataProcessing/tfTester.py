"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data #for the data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784]) # [a, b] Is the "shape" of the variable. None is a placeholder for any length (the number of input images), 784 is the number of pixels
  W = tf.Variable(tf.zeros([784, 10])) #Weight variable initialized a [784, 10] shape as zeroes so that it can be multiplied by x shape to produce vectors of length 10.
  b = tf.Variable(tf.zeros([10])) #Bias has a shape of 10 so we can add it to the output of W*x
  y = tf.matmul(x, W) + b #Matmul does matrix multiplication. These will be the predicted label scores.

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10]) #These will be the actual label scores.

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) #Calculates the error between y (predicted labels) and y_ (actual labels) using the cross entropy
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #Creates the gradient descent trainer that backpropogates attempting to reduce the cross-entropy error

  sess = tf.InteractiveSession() #initialize the training session
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) #Gets a random batch of 100 images and their corresponding labels
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) #the training happening, with every element in x corresponding to an image, and every element in y_ corresponding to its label

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #returns an array of booleans to weather the model predicted the correct number or not with tf.equal()
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #finds the accuracy by finding the mean of a "casted" set of bools, turning them into 1s and 0
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args( )
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)