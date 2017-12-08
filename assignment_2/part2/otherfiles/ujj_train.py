#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:31:32 2017

@author: anand
"""

# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-10-19

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

from dataset import TextDataset
from ujj_model import TextGenerationModel


def train(config):

    # Initialize the text dataset
    dataset = TextDataset(config.txt_file)

    # Initialize the model
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers
    )

    ###########################################################################
    # Implement code here.
    ###########################################################################

    X = tf.placeholder(dtype = tf.float32, shape = [config.batch_size, config.seq_length],name='inputs')
    Y = tf.placeholder(dtype = tf.int32, shape = [config.batch_size, config.seq_length],name='targets')
    Xp = tf.placeholder(dtype = tf.float32, shape = [None, None],name='inputs2')
    embedding_matrix = tf.eye(dataset.vocab_size)
    x_ohe = tf.nn.embedding_lookup(embedding_matrix,tf.cast(X,tf.int32))
    inputs = tf.nn.embedding_lookup(embedding_matrix,tf.cast(Xp,tf.int32))
    print (x_ohe.get_shape().as_list())
    probabilities, loss = model.probabilities(x_ohe,Y)
    predictions = model.predictions(inputs)

    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Compute the gradients for each variable
    grads_and_vars = optimizer.compute_gradients(loss)
    #train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)

    ###########################################################################
    # Implement code here.
    ###########################################################################
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for train_step in range(int(config.train_steps)):
        # Retreive training data batch
        batch_train, batch_target = dataset.batch(config.batch_size, config.seq_length)
        # Only for time measurement of step through network
        t1 = time.time()

        _, lossval = sess.run([apply_gradients_op,loss],
                                feed_dict={X: batch_train, Y: batch_target})

        #######################################################################
        # Implement code here.
        #######################################################################

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Output the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step+1,
                int(config.train_steps), config.batch_size, examples_per_second, lossval
            ))
            #print(dataset.convert_to_string(batch_target[0]))
            #print(pred[0])
            #print(dataset.convert_to_string(pred[0]))
            
        if(train_step%int(config.sample_every)==0):
            start = np.random.choice(np.arange(dataset.vocab_size),size=1)
            holder = []
            xv = [start]
            #for i in range(1,config.seq_length):
            holder.append(xv[0][0])
            while(len(holder)<=30):                
                pred = sess.run([predictions],feed_dict={Xp:xv})
                xv = pred
                holder.append(xv[0][0])
            print(holder)
            print(dataset.convert_to_string(holder))


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    #parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")

    # REMOVE THIS IN THE FINAL VERSION AND UNCOMMENT ABOVE. ONLY TO TEST.
    parser.add_argument('--txt_file', type=str, default = "./books/book_EN_grimms_fairy_tails.txt", help="Path to a .txt file to train on")

    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)