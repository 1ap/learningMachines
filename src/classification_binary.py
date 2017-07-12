#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
# author: Pavan P P ( pavanpadmashali@gmail.com)
"""
Provide a classifier that can classify binary inputs to binary outputs.
Input dimension : 2
Output dimension : 2
test system is an exclusive OR gate.
"""

import math
import tensorflow as tf
import numpy as np

HIDDEN_NODES = 10   # Number of hidden layers is one

x = tf.placeholder(tf.float32, [None, 2])  # inputs
W_hidden = tf.Variable(tf.truncated_normal([2, HIDDEN_NODES], stddev=1./math.sqrt(2)))  # initialize input to hidden layer weight
b_hidden = tf.Variable(tf.zeros([HIDDEN_NODES]))  # bias input
hidden = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)  # weighted sum available to hidden layer activation

W_logits = tf.Variable(tf.truncated_normal([HIDDEN_NODES, 2], stddev=1./math.sqrt(HIDDEN_NODES)))v # initialize hidden layer to output layer output
b_logits = tf.Variable(tf.zeros([2])) # bias input to output layer
logits = tf.matmul(hidden, W_logits) + b_logits  # weighted sum available at output layer

y = tf.nn.softmax(logits)  # ANN output

y_input = tf.placeholder(tf.float32, [None, 2])  # experimental output or Expected output

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,logits=y)
loss = tf.reduce_mean(cross_entropy)  # performance measure to be optimized. Reduce error.

train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)  # 0.2 is the learning rate

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

xTrain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # test input
yTrain = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # test output

for i in xrange(500):
  _, loss_val = sess.run([train_op, loss], feed_dict={x: xTrain, y_input: yTrain})

  if i % 10 == 0:
    print "Step:", i, "Current loss:", loss_val
    for x_input in [[0, 0], [0, 1], [1, 0], [1, 1]]:
      print x_input, sess.run(y, feed_dict={x: [x_input]})  # expected output should successively reduce error between expected output and network output
