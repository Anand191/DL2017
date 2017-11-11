"""
This module implements a convolutional neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l2_regularizer

class ConvNet(object):
  """
  This class implements a convolutional neural network in TensorFlow.
  It incorporates a certain graph model to be trained and to be used
  in inference.
  """

  def __init__(self, n_classes = 10):
    """
    Constructor for an ConvNet object. Default values should be used as hints for
    the usage of each parameter.
    Args:
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the ConvNet.
    """
    self.n_classes = n_classes
    self.weight_initializer = xavier_initializer()
    self.weight_regularizer = l2_regularizer(0.01)

  def inference(self, x):
    """
    Performs inference given an input tensor. This is the central portion
    of the network where we describe the computation graph. Here an input
    tensor undergoes a series of convolution, pooling and nonlinear operations
    as defined in this method. For the details of the model, please
    see assignment file.

    Here we recommend you to consider using variable and name scopes in order
    to make your graph more intelligible for later references in TensorBoard
    and so on. You can define a name scope for the whole model or for each
    operator group (e.g. conv+pool+relu) individually to group them by name.
    Variable scopes are essential components in TensorFlow for parameter sharing.
    Although the model(s) which are within the scope of this class do not require
    parameter sharing it is a good practice to use variable scope to encapsulate
    model.

    Args:
      x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
              the logits outputs (before softmax transformation) of the
              network. These logits can then be used with loss and accuracy
              to evaluate the model.
    """
    
    def conv_layer(ipt,shape,scope=None):
        with tf.variable_scope(scope or 'linear'):
            w = tf.get_variable(
                    'w',
                    shape,
                    initializer=self.weight_initializer,
                    regularizer = self.weight_regularizer
                )
            b = tf.get_variable(
                'b',
                shape[-1],
                initializer=self.weight_initializer
            )
            return(conv(ipt,w)+b)
            
    def linear(input, shape, scope=None, stddev=1.0):
        with tf.variable_scope(scope or 'linear'):
            w = tf.get_variable(
                'w1',
                shape,
                initializer=self.weight_initializer,
                regularizer = self.weight_regularizer
            )
            b = tf.get_variable(
                'b1',
                shape[-1],
                initializer=self.weight_initializer
            )
            return tf.matmul(input, w) + b
            
    
    def conv(ipt,W):
        return(tf.nn.conv2d(ipt,W,strides=[1,1,1,1],padding='SAME'))
    
    def max_pool(ipt):
        return(tf.nn.max_pool(ipt, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME'))
        
# =============================================================================
#     def weight_variable(shape):
#         initial = tf.truncated_normal(shape, stddev=0.1)
#         return tf.Variable(initial)
#     
#     def bias_variable(shape):
#         initial = tf.constant(0.1, shape=shape)
#         return tf.Variable(initial)
# =============================================================================
    
    with tf.name_scope('layer1'):
# =============================================================================
#         W_conv1 = weight_variable([5, 5, 3, 64])
#         b_conv1 = bias_variable([64])
#         h_conv1 = tf.nn.relu(conv(x,W_conv1)+b_conv1)
# =============================================================================
        h_conv1 = tf.nn.relu(conv_layer(x,[5, 5, 3, 64],'c1'))
        h_pool1 = max_pool(h_conv1)
        
    with tf.name_scope('layer2'):
# =============================================================================
#         W_conv2 = weight_variable([5, 5, 64, 64])
#         b_conv2 = bias_variable([64])
#         h_conv2 = tf.nn.relu(conv(h_pool1,W_conv2)+b_conv2)
# =============================================================================
        h_conv2 = tf.nn.relu(conv_layer(h_pool1,[5, 5, 64, 64],'c2'))
        h_pool2 = max_pool(h_conv2)

    with tf.name_scope('flatten'):
        h_flat = tf.reshape(h_pool2,[-1,8*8*64])
    
    with tf.name_scope('f1'):
# =============================================================================
#         W_f1 = weight_variable([8*8*64,384])
#         b_f1 = bias_variable([384])
#         h_f1 = tf.nn.relu(tf.matmul(h_flat, W_f1) + b_f1)
# =============================================================================
        h_f1 = tf.nn.relu(linear(h_flat,[8*8*64,384],'fc1'))
        
    with tf.name_scope('f2'):         
# =============================================================================
#         W_f2 = weight_variable([384,192])
#         b_f2 = bias_variable([192])
#         h_f2 = tf.nn.relu(tf.matmul(h_f1, W_f2) + b_f2)
# =============================================================================
        h_f2 = tf.nn.relu(linear(h_f1,[384,192],'fc2'))
        
    with tf.name_scope('f3'):
# =============================================================================
#         W_f3 = weight_variable([192,self.n_classes])
#         b_f3 = bias_variable([self.n_classes])
#         logits = tf.matmul(h_f2, W_f3) + b_f3   
# =============================================================================
        logits = linear(h_f2,[192,self.n_classes],'fout')
        return logits

  def loss(self, logits, labels):
    """
    Calculates the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    In order to implement this function you should have a look at
    tf.nn.softmax_cross_entropy_with_logits.
    
    You can use tf.summary.scalar to save scalar summaries of
    cross-entropy loss, regularization loss, and full loss (both summed)
    for use with TensorBoard. This will be useful for compiling your report.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.

    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    loss = tf.reduce_mean(cross_entropy)

    return loss

  def train_step(self, loss, flags):
    """
    Implements a training step using a parameters in flags.

    Args:
      loss: scalar float Tensor.
      flags: contains necessary parameters for optimization.
    Returns:
      train_step: TensorFlow operation to perform one training step
    """
    train_step = tf.train.AdamOptimizer(learning_rate=flags).minimize(loss)

    return train_step

  def accuracy(self, logits, labels):
    """
    Calculate the prediction accuracy, i.e. the average correct predictions
    of the network.
    As in self.loss above, you can use tf.scalar_summary to save
    scalar summaries of accuracy for later use with the TensorBoard.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.

    Returns:
      accuracy: scalar float Tensor, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """
    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    return accuracy

