#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:34:07 2017

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

import tensorflow as tf

class TextGenerationModel(object):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size
        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)

        # Declare LSTM Structure
        self._model_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(self._lstm_num_hidden) for _ in range(self._lstm_num_layers)])

        # Initialize Output Softmax Structure
        self.softmax_weights = tf.get_variable("Wout", [self._lstm_num_hidden, self._vocab_size],
                                         initializer=initializer_weights, dtype=tf.float32)
        self.softmax_biases = tf.get_variable("bout", [self._vocab_size],
                                         initializer=initializer_biases, dtype=tf.float32)

    def _build_model(self,x):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]

        # Use a RNN Cell Wrapper method to initialize all the input state variables to zero.
        #state = self._model_cell.zero_state(self._batch_size, tf.float32)

        lstm_outputs, state = tf.nn.dynamic_rnn(self._model_cell, x, dtype=tf.float32)

        lstm_reshaped_outputs = tf.reshape(lstm_outputs, [-1, self._lstm_num_hidden])
        print (lstm_reshaped_outputs.get_shape().as_list())

        logits_per_step = tf.add(tf.matmul(lstm_reshaped_outputs, self.softmax_weights),self.softmax_biases)
        logits_per_step = tf.reshape(logits_per_step, [-1, self._seq_length, self._vocab_size])
        print (logits_per_step.get_shape().as_list())

        return logits_per_step, state

    def _compute_loss(self,logits,targets):
        # Cross-entropy loss, averaged over timestep and batch
        weights = tf.ones([self._batch_size, self._seq_length])
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=targets,weights=weights)
        return loss

    def probabilities(self,x,targets):
        # Returns the normalized per-step probabilities
        logits_per_step, state = self._build_model(x)
        loss = self._compute_loss(logits_per_step,targets)
        probabilities = tf.nn.softmax(logits_per_step)
        print (probabilities.get_shape().as_list())
        return probabilities, loss

    def predictions(self,x):
        # Returns the per-step predictions
        outputs,states = tf.nn.dynamic_rnn(self._model_cell,x,dtype=tf.float32)
        outputs_reshaped = tf.reshape(outputs, [-1, self._lstm_num_hidden])
        print (outputs_reshaped.get_shape().as_list())
        logits = tf.matmul(outputs_reshaped,self.softmax_weights) + self.softmax_biases
        print (logits.get_shape().as_list())
        predictions = tf.argmax(logits,axis=1)
        return predictions