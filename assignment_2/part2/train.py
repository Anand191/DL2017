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
from model import TextGenerationModel


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

    initial_state = tf.placeholder(shape=[config.lstm_num_layers,2,None,config.lstm_num_hidden],dtype=tf.float32)
    x = tf.placeholder(shape=[None,None],dtype=tf.int32)
    y = tf.placeholder(shape=[None,None],dtype=tf.int32)
 
    vocab_size = dataset.vocab_size
    em_matrix =  tf.get_variable("embedding", [vocab_size, vocab_size],initializer = tf.initializers.identity) # tf.eye(dataset.vocab_size)
    inputs = tf.nn.embedding_lookup(em_matrix, x)
    print ("input shape={}".format(inputs.get_shape().as_list()))
    
    probabilities,logits,n_states,loss,accuracy = model.probabilities(inputs,y,initial_state)
    prediction = model.predictions(probabilities)


    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate)

    # Compute the gradients for each variable
    grads_and_vars = optimizer.compute_gradients(loss)
    global_step = tf.Variable(0,trainable=False)
    #train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)


    #saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_f = tf.summary.FileWriter(config.summary_path,sess.graph)
    s_acc_train = tf.summary.scalar('training_acc',accuracy)
    s_loss_train = tf.summary.scalar('train_loss',loss)

    for train_step in range(int(config.train_steps)):
        xt,yt = dataset.batch(config.batch_size, config.seq_length)        

        # Only for time measurement of step through network
        t1 = time.time()  


        # sess.run ( .. )
        _,ls,acc,sltr,satr = sess.run([apply_gradients_op,loss,accuracy,s_loss_train,s_acc_train],
                              feed_dict={x:xt,y:yt,initial_state:np.zeros((config.lstm_num_layers,2,
                                                                           config.batch_size,config.lstm_num_hidden))})
        summary_f.add_summary(sltr,train_step)
        summary_f.add_summary(satr,train_step)

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Output the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {},Accuracy = {}"
                  .format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step+1,
                int(config.train_steps), config.batch_size, examples_per_second,ls,acc
            ))
            
        if(train_step%int(config.sample_every)==0):
            start = np.random.choice(np.arange(dataset.vocab_size),size=1)
            start2 = np.reshape(start,(1,1))
            sent = []
            state = np.zeros((config.lstm_num_layers,2,1,config.lstm_num_hidden))
            for i in range(1,60):
                pred,state = sess.run([prediction,n_states],feed_dict={x:start2,initial_state:state})
                sent.append(pred[0])
                start2[0] = pred
            print("Sentence:{}".format(dataset.convert_to_string(sent)))
              


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str,default="./books/book_NL_darwin_reis_om_de_wereld.txt", help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=15000, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()
    # Train the model
    train(config)