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

import numpy as np
import tensorflow as tf

################################################################################

class VanillaRNN(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)
        
        self.U = tf.get_variable('U', shape=[self._input_dim, self._num_hidden], 
                    initializer=initializer_weights)
        self.b = tf.get_variable('bi', shape=[self._num_hidden], 
                        initializer=initializer_biases)
        
        self.W = tf.get_variable('W', shape=[self._num_hidden, self._num_hidden], 
                    initializer=initializer_weights)
        
        
        self.V = tf.get_variable('V', shape=[self._num_hidden, self._num_classes], 
                    initializer=initializer_weights)
        self.bo = tf.get_variable('bo', shape=[self._num_classes], 
                        initializer=initializer_biases)
        
        

        # Initialize the stuff you need
        # ...

    def _rnn_step(self, h_prev, x):
        # Single step through Vanilla RNN cell ...
        return tf.tanh((tf.matmul(x,self.U) + self.b) + tf.matmul(h_prev, self.W))
        
    def compute_logits(self,ipt,init_state):
        # Implement the logits for predicting the last digit in the palindrome
        def output(hidden_state):
            return(tf.matmul(hidden_state,self.V)+self.bo)
        inp = tf.transpose(ipt,perm=[1,0,2])
        states = tf.scan(self._rnn_step,inp,initializer=init_state)
        #states_reshaped = tf.reshape(states, [-1, self._num_hidden])
        #logits = tf.matmul(states_reshaped, self.V) + self.bo
        logits_all = tf.map_fn(output,states)
        logits = logits_all[-1]
        return logits

    def compute_loss(self,logits,labels):
        # Implement the cross-entropy loss for classification of the last digit
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
        return loss

    def accuracy(self,logits,labels):
        # Implement the accuracy of predicting the
        # last digit over the current batch ...
        correct = tf.equal(tf.to_int32(tf.argmax(logits,1)), labels)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy
