from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'


def train():
  """
  Performs training and evaluation of ConvNet model.

  First define your graph using class ConvNet and its methods. Then define
  necessary operations such as savers and summarizers. Finally, initialize
  your model within a tf.Session and do the training.

  ---------------------------
  How to evaluate your model:
  ---------------------------
  Evaluation on test set should be conducted over full batch, i.e. 10k images,
  while it is alright to do it over minibatch for train set.

  ---------------------------------
  How often to evaluate your model:
  ---------------------------------
  - on training set every print_freq iterations
  - on test set every eval_freq iterations

  ------------------------
  Additional requirements:
  ------------------------
  Also you are supposed to take snapshots of your model state (i.e. graph,
  weights and etc.) every checkpoint_freq iterations. For this, you should
  study TensorFlow's tf.train.Saver class.
  """

  # Set the random seeds for reproducibility. DO NOT CHANGE.
  tf.set_random_seed(42)
  np.random.seed(42)
  
  from convnet_tf import ConvNet
  import cifar10_utils
  
  cifar10 = cifar10_utils.get_cifar10()
  
  #n_in = 32*32*3
  n_out = 10
  tbatch = 400
  
  cn = ConvNet()
  x = tf.placeholder(tf.float32, [None, 32,32,3])
  y = tf.placeholder(tf.float32, [None, n_out])
  
  logits = cn.inference(x)
  cross_loss = cn.loss(logits,y)
  optimizer = cn.train_step(cross_loss,LEARNING_RATE_DEFAULT)
  acc = cn.accuracy(logits,y)
  
  Xval,Yval = cifar10.train.next_batch(5000)
  Xt, Yt = cifar10.test.images, cifar10.test.labels
  Xt1, Yt1 = Xt[0:5000], Yt[0:5000]
  Xt2, Yt2 = Xt[5000:], Yt[5000:]
  
  #Xtr = np.reshape(Xtr,(Xtr.shape[0],n_in))
   
  with tf.Session() as sess:
      tf.local_variables_initializer().run()
      tf.global_variables_initializer().run()
      for epoch in range(25):
          #avg_cost = 0.
          #avg_acc = 0.
          print ("Begin Epoch {}".format(epoch+1))
          for i in range(tbatch):
              X, Y = cifar10.train.next_batch(BATCH_SIZE_DEFAULT)
              #X = np.reshape(X,(X.shape[0],n_in))          
              opt,ls,train_acc = sess.run([optimizer,cross_loss,acc],feed_dict={x: X, y: Y})

              #avg_cost += ls/tbatch
              #avg_acc += train_acc/tbatch
              if(i%100==0):
                  print("No. of steps remaining in this epoch = {}".format(tbatch-i))
          
          avg_acc = acc.eval(feed_dict={x: Xval, y: Yval})
          if(epoch%1==0):
              print('step %d,training accuracy =  %g' % (epoch+1, avg_acc))
      print("training finished!!!")
      bat1_acc = acc.eval(feed_dict={x: Xt1, y:Yt1})
      bat2_acc = acc.eval(feed_dict={x: Xt2, y:Yt2})
      
      print('test accuracy - batch1 %g' % bat1_acc)
      print('test accuracy - batch2 %g' % bat2_acc)
      print ('test accuracy -  total = %g' % ((bat1_acc+bat2_acc)/2))
      


def initialize_folders():
  """
  Initializes all folders in FLAGS variable.
  """

  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)

  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  print_flags()

  initialize_folders()

  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
  parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
  parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
  parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
  parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
