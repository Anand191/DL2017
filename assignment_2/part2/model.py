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

        # Initialization:
        # ...
        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)
        
        self.V = tf.get_variable('V', shape=[self._lstm_num_hidden, self._vocab_size], 
                    initializer=initializer_weights)
        self.bo = tf.get_variable('bo', shape=[self._vocab_size], 
                        initializer=initializer_biases)   
        
        self.lstm_cells = [tf.contrib.rnn.BasicLSTMCell(self._lstm_num_hidden) for i in range(self._lstm_num_layers)]
        self.rnn_model = tf.contrib.rnn.MultiRNNCell(self.lstm_cells)
        


    def _build_model(self,ipt,init_state):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]
        state_u = tf.unstack(init_state)
        init = tuple([tf.nn.rnn_cell.LSTMStateTuple(state_u[i][0],state_u[i][1])for i in range(self._lstm_num_layers)])
        outputs, n_states = tf.nn.dynamic_rnn(self.rnn_model,ipt,initial_state=init)
        outputs_reshaped = tf.reshape(outputs, [-1,self._lstm_num_hidden])
        
        logits = tf.matmul(outputs_reshaped, self.V) + self.bo
        logits_per_step = tf.reshape(logits,(-1,self._seq_length,self._vocab_size))
        print ("logits shape = {}".format(logits_per_step.get_shape().as_list()))
        return logits_per_step,n_states

    def _compute_loss(self,logits,y):
        # Cross-entropy loss, averaged over timestep and batch
        wt = tf.ones([self._batch_size,self._seq_length])
        loss = tf.contrib.seq2seq.sequence_loss(logits,y,weights=wt)
        return loss

    def probabilities(self,x,y,init_state):
        # Returns the normalized per-step probabilities
        logits,n_states = self._build_model(x,init_state)
        loss = self._compute_loss(logits,y)
        correct = tf.equal(tf.to_int32(tf.argmax(logits,2)), y)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        logits = tf.reshape(logits,(-1,self._vocab_size))
        
        probabilities = tf.nn.softmax(logits)
        print ("probabilities shape={}".format(probabilities.get_shape().as_list()))
        return probabilities,logits,n_states,loss,accuracy

    def predictions(self,probabilities):
        # Returns the per-step predictions
        predictions = tf.argmax(probabilities,axis=1)
        print ("predictions shape={}".format(predictions.get_shape().as_list()))
        return predictions