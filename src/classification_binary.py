#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
#
# author: Pavan P P ( pavanpadmashali@gmail.com)
"""
Provide a classifier that can classify binary inputs to binary outputs.
Input dimension : 2
Output dimension : 2
test system is an exclusive OR gate.
"""

import tensorflow as tf
import numpy as np

HIDDEN_NODES = 10   # Number of hidden layers is one

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(HIDDEN_NODES, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.2),
              loss='binary_crossentropy')

model.fit(x_train, y_train, epochs=50, verbose=2)


for x_input in [[0, 0], [0, 1], [1, 0], [1, 1]]:
  prediction = model.predict(np.array([x_input]))
  binary_output = 1 if prediction > 0.5 else 0
  print(f"Input: {x_input}, Predicted Output: {binary_output}")