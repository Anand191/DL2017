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


class LSTM(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)

        # Initialize the stuff you need
        # ...
        
        self.W_gx = tf.get_variable('W_gx', shape=[self._input_dim, self._num_hidden], 
                    initializer=initializer_weights)
        self.W_gh = tf.get_variable('W_gh', shape=[self._num_hidden, self._num_hidden], 
                    initializer=initializer_weights)
        self.b_g = tf.get_variable('b_g', shape=[self._num_hidden], 
                        initializer=initializer_biases)
        
        self.W_ix = tf.get_variable('W_ix', shape=[self._input_dim, self._num_hidden], 
                    initializer=initializer_weights)
        self.W_ih = tf.get_variable('W_ih', shape=[self._num_hidden, self._num_hidden], 
                    initializer=initializer_weights)
        self.b_i = tf.get_variable('b_i', shape=[self._num_hidden], 
                        initializer=initializer_biases)
        
        self.W_fx = tf.get_variable('W_fx', shape=[self._input_dim, self._num_hidden], 
                    initializer=initializer_weights)
        self.W_fh = tf.get_variable('W_fh', shape=[self._num_hidden, self._num_hidden], 
                    initializer=initializer_weights)
        self.b_f = tf.get_variable('b_f', shape=[self._num_hidden], 
                        initializer=initializer_biases)
        
        self.W_ox = tf.get_variable('W_ox', shape=[self._input_dim, self._num_hidden], 
                    initializer=initializer_weights)
        self.W_oh = tf.get_variable('W_oh', shape=[self._num_hidden, self._num_hidden], 
                    initializer=initializer_weights)
        self.b_o = tf.get_variable('b_o', shape=[self._num_hidden], 
                        initializer=initializer_biases)       
        
        
        self.V = tf.get_variable('W_out', shape=[self._num_hidden, self._num_classes], 
                    initializer=initializer_weights)
        self.bout = tf.get_variable('b_out', shape=[self._num_classes], 
                        initializer=initializer_biases)

    def _lstm_step(self, lstm_state_tuple, x):
        # Single step through LSTM cell ...
        h,c = tf.unstack(lstm_state_tuple)
        g = tf.tanh(tf.matmul(x,self.W_gx) + tf.matmul(h,self.W_gh) + self.b_g)
        i = tf.sigmoid((tf.matmul(x,self.W_ix) + tf.matmul(h,self.W_ih) + self.b_i))
        f = tf.sigmoid((tf.matmul(x,self.W_fx) + tf.matmul(h,self.W_fh) + self.b_f))
        o = tf.sigmoid((tf.matmul(x,self.W_ox) + tf.matmul(h,self.W_oh) + self.b_o))
        
        ct = tf.multiply(g,i) + tf.multiply(c,f)
        ht = tf.multiply(tf.tanh(ct),o)
        
        return (tf.stack([ht,ct]))
    
    def compute_logits(self,ipt,init_state):
        # Implement the logits for predicting the last digit in the palindrome
        print (init_state)
        def output(hidden_state):
            return(tf.matmul(hidden_state,self.V)+self.bout)
        inp = tf.transpose(ipt,perm=[1,0,2])
        states = tf.scan(self._lstm_step,inp,initializer=init_state)
        print(states)
        states_reshape = tf.transpose(states,perm=[1,0,2,3])
        states_h = states_reshape[0]
        print(states_h)
        
        logits_all = tf.map_fn(output,states_h)
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