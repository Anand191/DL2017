"""
This module implements a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in Tensorflow.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference, training and it
  can also be used for evaluating prediction performance.
  """

  def __init__(self, n_hidden, n_classes, is_training,
               activation_fn = tf.nn.relu, dropout_rate = 0.,
               weight_initializer = xavier_initializer(),
               weight_regularizer = l2_regularizer(0.001)):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter.

    Args:
      n_hidden: list of ints, specifies the number of units
                     in each hidden layer. If the list is empty, the MLP
                     will not have any hidden units, and the model
                     will simply perform a multinomial logistic regression.
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the MLP.
      is_training: Bool Tensor, it indicates whether the model is in training
                        mode or not. This will be relevant for methods that perform
                        differently during training and testing (such as dropout).
                        Have look at how to use conditionals in TensorFlow with
                        tf.cond.
      activation_fn: callable, takes a Tensor and returns a transformed tensor.
                          Activation function specifies which type of non-linearity
                          to use in every hidden layer.
      dropout_rate: float in range [0,1], presents the fraction of hidden units
                         that are randomly dropped for regularization.
      weight_initializer: callable, a weight initializer that generates tensors
                               of a chosen distribution.
      weight_regularizer: callable, returns a scalar regularization loss given
                               a weight variable. The returned loss will be added to
                               the total loss for training purposes.
    """
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.is_training = is_training
    self.activation_fn = activation_fn
    self.dropout_rate = dropout_rate
    self.weight_initializer = weight_initializer
    self.weight_regularizer = weight_regularizer

  def inference(self, x):
    """
    Performs inference given an input tensor. This is the central portion
    of the network. Here an input tensor is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    In order to keep things uncluttered we recommend you (though it's not required)
    to implement a separate function that is used to define a fully connected
    layer of the MLP.

    In order to make your code more structured you can use variable scopes and name
    scopes. You can define a name scope for the whole model, for each hidden
    layer and for output. Variable scopes are an essential component in TensorFlow
    design for parameter sharing.

    You can use tf.summary.histogram to save summaries of the fully connected layer weights,
    biases, pre-activations, post-activations, and dropped-out activations
    for each layer. It is very useful for introspection of the network using TensorBoard.

    Args:
      x: 2D float Tensor of size [batch_size, input_dimensions]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """
    #n_input = x.get_shape()[1].value
    
    def linear(input, output_dim, scope=None, stddev=1.0):
        with tf.variable_scope(scope or 'linear'):
            w = tf.get_variable(
                'w',
                [input.get_shape()[1], output_dim],
                initializer=tf.random_normal_initializer(stddev=stddev)
            )
            b = tf.get_variable(
                'b',
                [output_dim],
                initializer=tf.constant_initializer(0.0)
            )
            return tf.matmul(input, w) + b
        
    def fcn(input,h_dim,out_dim):
        h0 = tf.nn.relu(linear(input,h_dim[0],'n0'))
        h1 = tf.nn.relu(linear(h0,h_dim[1],'n1'))
        out = linear(h1,out_dim,'nout')
        return(out)
        
# =============================================================================
#     def weight_var(shape):
#         initial = tf.truncated_normal(shape, stddev=0.1)
#         return tf.Variable(initial)
#     
#     def bias_var(shape):
#         #initial = tf.constant(0.1, shape=shape)
#         initial = tf.random_normal(shape, stddev=0.1)
#         return tf.Variable(initial)
#     
#     def fc(x,w,b):
#         return (tf.add(tf.matmul(x, w),b))
# =============================================================================
# =============================================================================
#     def mlp(x,weights,biases):
#         layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#         layer_1 = tf.nn.relu(layer_1)
#         
#         layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#         layer_2 = tf.nn.relu(layer_2)
#         
#         out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#         
#         return (out_layer)       
#         
#         
#     n_input = x.get_shape()[1].value
#     weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, self.n_hidden[0]])),
#     'h2': tf.Variable(tf.random_normal([self.n_hidden[0], self.n_hidden[1]])),
#     'out': tf.Variable(tf.random_normal([self.n_hidden[1], self.n_classes]))
#     }
#     biases = {
#     'b1': tf.Variable(tf.random_normal([self.n_hidden[0]])),
#     'b2': tf.Variable(tf.random_normal([self.n_hidden[1]])),
#     'out': tf.Variable(tf.random_normal([self.n_classes]))
#     }
# =============================================================================

# =============================================================================
#     w1 = weight_var([n_input,self.n_hidden[0]])
#     b1 = bias_var([self.n_hidden[0]])
#     layer_1 = fc(x,w1,b1)
#     layer_1 = tf.nn.relu(layer_1)
#         
# 
#     w2 = weight_var([self.n_hidden[0],self.n_hidden[1]])
#     b2 = bias_var([self.n_hidden[1]])
#     layer_2 = fc(layer_1,w2,b2)
#     layer_2 = tf.nn.relu(layer_2)
#     
#      
#     w_o = weight_var([self.n_hidden[1],self.n_classes])
#     b_o = bias_var([self.n_classes])
#     logits = fc(layer_2,w_o,b_o)
# =============================================================================
    logits = fcn(x,self.n_hidden,self.n_classes)

    return logits

  def loss(self, logits, labels):
    """
    Computes the multiclass cross-entropy loss from the logits predictions and
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

# =============================================================================
#     ########################
#     # PUT YOUR CODE HERE  #
#     #######################
#     raise NotImplementedError
#     ########################
#     # END OF YOUR CODE    #
#     #######################
# =============================================================================

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

# =============================================================================
#     ########################
#     # PUT YOUR CODE HERE  #
#     #######################
#     raise NotImplementedError
#     ########################
#     # END OF YOUR CODE    #
#     #######################
# =============================================================================

    return train_step

  def accuracy(self, logits, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    As in self.loss above, you can use tf.summary.scalar to save
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

# =============================================================================
#     ########################
#     # PUT YOUR CODE HERE  #
#     #######################
#     raise NotImplementedError
#     ########################
#     # END OF YOUR CODE    #
#     #######################
# 
# =============================================================================
    return accuracy
