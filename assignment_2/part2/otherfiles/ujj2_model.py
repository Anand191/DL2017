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
import numpy as np

class TextGenerationModel(object):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases = tf.constant_initializer(0.0)

        # Declare LSTM Structure
        self._model_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(self._lstm_num_hidden) for _ in range(self._lstm_num_layers)])

        # Initialize Output Softmax Structure
        self.softmax_weights = tf.get_variable("Wout", [self._lstm_num_hidden, self._vocab_size],
                                         initializer=initializer_weights, dtype=tf.float32)
        self.softmax_biases = tf.get_variable("bout", [self._vocab_size],
                                         initializer=initializer_biases, dtype=tf.float32)


    def _build_model(self, x, train, initial_state):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]

        # logits_per_step  = [batch_size, seq_len, vocab_size]

        input_seq_length = tf.where(train, self._seq_length, 1)

        # Taken from https://stackoverflow.com/questions/39112622/how-do-i-set-tensorflow-rnn-state-when-state-is-tuple-true
        state_per_layer_list = tf.unstack(initial_state, axis=0)
        init_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(self._lstm_num_layers)]
        )

        lstm_outputs, state = tf.nn.dynamic_rnn(self._model_cell, x, initial_state=init_state)

        # Use a RNN Cell Wrapper method to initialize all the input state variables to zero.
        #initial_state = self._model_cell.zero_state(self._batch_size, tf.float32)
        #lstm_outputs, state = tf.nn.dynamic_rnn(self._model_cell, self.x_ohe, initial_state=initial_state)
        lstm_reshaped_outputs = tf.reshape(lstm_outputs, [-1, self._lstm_num_hidden])
        logits_per_step = tf.add(tf.matmul(lstm_reshaped_outputs, self.softmax_weights),self.softmax_biases)
        logits_per_step = tf.reshape(logits_per_step, [-1, input_seq_length, self._vocab_size])
        return logits_per_step, state

    def _compute_loss(self,logits,labels):
        # Cross-entropy loss, averaged over timestep and batch
        weights = tf.ones([self._batch_size, self._seq_length])
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=labels,weights=weights)
        return loss

    def probabilities(self, x, train, labels, init_state):
        # Returns the normalized per-step probabilities
        #logits_per_step, state = self._build_model(x)
        #loss = self._compute_loss(logits_per_step,targets)
        logits_per_step, state = self._build_model(x, train, init_state)
        loss = self._compute_loss(logits_per_step,labels)
        logits = tf.reshape(logits_per_step, [-1, self._vocab_size])
        probabilities = tf.nn.softmax(logits)
        return probabilities, loss, state, logits

    # def sample(self):
    #     # if self.init is True:
    #     #     self.sampling_state = self._model_cell.zero_state(1, tf.float32)
    #     l = tf.unstack(self.sampling_state, axis=0)
    #     state = tuple(
    #         [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
    #          for idx in range(self._lstm_num_layers)]
    #     )
    #     input_ohe = tf.nn.embedding_lookup(self.embedding_matrix, tf.cast(self.X_sampling, tf.int32))
    #     lstm_outputs, state = tf.nn.dynamic_rnn(self._model_cell, input_ohe, initial_state=state, dtype= tf.float32)
    #     lstm_reshaped_outputs = tf.reshape(lstm_outputs, [-1, self._lstm_num_hidden])
    #     logits_per_step = tf.add(tf.matmul(lstm_reshaped_outputs, self.softmax_weights), self.softmax_biases)
    #     logits_per_step = tf.reshape(logits_per_step, [-1, 1, self._vocab_size])
    #     predicted_char = tf.argmax(logits_per_step, axis=2)
    #     #pred = np.transpose(pred)
    #     return predicted_char, state

    def predictions(self,probabilities):
        #Returns the per-step predictions
        predictions = tf.argmax(probabilities, axis=1)
        return predictions

    # def sample(self, probailities, priming_word):



