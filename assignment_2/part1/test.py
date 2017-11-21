#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:46:17 2017

@author: anand
"""

import numpy as np

from sklearn import datasets
from sklearn.cross_validation import train_test_split
import tensorflow as tf

sess = tf.Session()
target_size = 10
digits = datasets.load_digits()
X = digits.images
Y_ = digits.target
Y = sess.run(tf.one_hot(indices=Y_, depth=target_size))
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.22, random_state=42)