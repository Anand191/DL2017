import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_mnist_images(binarize=True):
    """
    :param binarize: Turn the images into binary vectors
    :return: x_train, x_test  Where
        x_train is a (55000 x 784) tensor of training images
        x_test is a  (10000 x 784) tensor of test images
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    x_train = mnist.train.images
    x_test = mnist.test.images
    if binarize:
        x_train = (x_train>0.5).astype(x_train.dtype)
        x_test = (x_test>0.5).astype(x_test.dtype)
    return x_train, x_test

#%%
class NaiveBayesModel(object):

    def __init__(self,n_labels, w_init, b_init = None, c_init = None):
        """
        :param w_init: An (n_categories, n_dim) array, where w[i, j] represents log p(X[j]=1 | Z[i]=1)
        :param b_init: A (n_categories, ) vector where b[i] represents log p(Z[i]=1), or None to fill with zeros
        :param c_init: A (n_dim, ) vector where b[j] represents log p(X[j]=1), or None to fill with zeros
        """
        self.w = w_init
        self.b = b_init
        self.c = c_init
        self.n_labels = n_labels

    def log_p_x_given_z(self, x, z):
        """
        :param x: An (n_samples, n_dims) tensor
        :param z: An (n_samples, n_labels) tensor of integer class labels
        :return: An (n_samples, n_labels) tensor  p_x_given_z where result[i, j] indicates p(X=x[i] | Z=z[j])
        """
        param1 = tf.log_sigmoid(tf.transpose(tf.add(self.w, self.c)))
        param2 = tf.log_sigmoid(tf.transpose(tf.add(-self.w, -self.c)))
        return (tf.matmul(x,param1) + tf.matmul(tf.add(-x,1.),param2))        
        

    def log_p_x(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of log-probabilities assigned to each point
        """
        p_z_k = tf.log(tf.nn.softmax(self.b))
        p_x_z = self.log_p_x_given_z(x,None)
        return tf.reduce_logsumexp(tf.add(p_z_k,p_x_z),axis=1)

    def sample(self, n_samples):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """
        mu_d = tf.sigmoid(self.w+self.c)
        z = np.random.choice(self.n_labels,n_samples)
        X = np.zeros((n_samples,self.w.get_shape()[1]))
        for i in range(n_samples):
            X[i] = tf.distributions.Bernoulli(probs=mu_d[z[i]]).sample().eval()
        print(X.shape)
        plt.figure()
        for j in range(X.shape[0]):
            plt.subplot(4,4,j+1)
            plt.imshow(np.reshape(X[j],(28,28)))
            plt.axis('off')
        plt.savefig('Sample.png')
        plt.close()
        
    def sample_k(self, n_samples):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """
        mu_d = tf.sigmoid(self.w+self.c)
        z = np.random.choice(self.n_labels,n_samples)
        Z_K = np.zeros((n_samples,self.w.get_shape()[1]))
        for i in range(n_samples):
            Z_K[i] = mu_d[z[i]].eval()
        print(Z_K.shape)
        plt.figure()
        for j in range(Z_K.shape[0]):
            plt.subplot(5,4,j+1)
            plt.imshow(np.reshape(Z_K[j],(28,28)))
            plt.axis('off')
        plt.savefig('SampleZ.png')
        plt.close()

#%%
def train_simple_generative_model_on_mnist(n_categories=20, initial_mag = 0.01, optimizer='rmsprop', learning_rate=.01, n_epochs=20, test_every=100,
                                           minibatch_size=100, plot_n_samples=16):
    """
    Train a simple Generative model on MNIST and plot the results.

    :param n_categories: Number of latent categories (K in assignment)
    :param initial_mag: Initial weight magnitude
    :param optimizer: The name of the optimizer to use
    :param learning_rate: Learning rate for the optimization
    :param n_epochs: Number of epochs to train for
    :param test_every: Test every X iterations
    :param minibatch_size: Number of samples in a minibatch
    :param plot_n_samples: Number of samples to plot
    """

    # Get Data
    summary_path = "./summaries2"
    
    x_train, x_test = load_mnist_images(binarize=True)
    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors
    
    test_iterator = tf.data.Dataset.from_tensor_slices(x_test).repeat().batch(minibatch_size).make_initializable_iterator()
    t_minibatch = test_iterator.get_next()
    
    kernel_init = tf.truncated_normal_initializer(0, initial_mag)
    w_init = tf.get_variable(
                'w',
                [n_categories, n_dims],
                initializer=kernel_init,
            )
    c_init = tf.get_variable(
                'c',
                [n_dims],
                initializer=kernel_init
            )
    
    b_init = tf.get_variable(
                'b',
                [n_categories],
                initializer=kernel_init
            )

    # Build the model
    model = NaiveBayesModel(n_categories,w_init,b_init,c_init)
    loss = tf.reduce_mean(model.log_p_x(x_minibatch))
    loss_t = tf.reduce_mean(model.log_p_x(t_minibatch))
    
    optim = tf.train.RMSPropOptimizer(learning_rate).minimize(-loss)
# =============================================================================
#     grads_and_vars = optim.compute_gradients(loss)
#     global_step = tf.Variable(0,trainable=False)
#     grads, variables = zip(*grads_and_vars)
#     apply_gradients_op = optim.apply_gradients(zip(grads, variables),global_step=global_step) 
# =============================================================================

    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        
        summary_f = tf.summary.FileWriter(summary_path,sess.graph)
        s_loss_train = tf.summary.scalar('train_log_prob',loss)
        s_loss_test = tf.summary.scalar('test_log_prob',loss_t)
        n_steps = (n_epochs * n_samples)/minibatch_size
        for i in range(int(n_steps)):
            opt,ls,sltr = sess.run([optim,loss,s_loss_train])
            summary_f.add_summary(sltr,i)
            
            if i%test_every==0:
                test_loss,slts = sess.run([loss_t,s_loss_test])
                summary_f.add_summary(slts,i)
                print("Train Step {:04d}/{:04d}, Batch Size = {}, Train Log_Prob = {}, Test Log_Prob = {}"
                      .format(i + 1,int(n_steps), minibatch_size, ls, test_loss)) 
                

        model.sample(plot_n_samples)
        model.sample_k(n_categories)
        
        #x_n = tf.placeholder(tf.float32, [None, n_dims])
        #x_f = tf.placeholder(tf.float32, [None, n_dims])
        
        idx = np.random.choice(x_train.shape[0],10)
        Normal = x_train[idx]
        h1,h2 = np.array_split(Normal,2,axis=1)
        h2 = np.roll(h2,1,axis=0)
        Frankenstein = np.hstack((h1,h2))
        
        loss_n = -model.log_p_x(Normal)
        loss_f = -model.log_p_x(Frankenstein)
        
        normal_loss, frankenstein_loss = sess.run([loss_n,loss_f])
        normal_loss, frankenstein_loss = -normal_loss, -frankenstein_loss
        
        plt.figure()
        for j in range(10):
            plt.subplot(2,5,j+1)
            plt.imshow(np.reshape(Normal[j],(28,28)))
            plt.title(normal_loss[j])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('Normal.png')
        plt.close()
        
        plt.figure()
        for j in range(10):            
            plt.subplot(2,5,j+1)
            plt.imshow(np.reshape(Frankenstein[j],(28,28)))
            plt.title(frankenstein_loss[j])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('Frankenstein.png')
        plt.close()
        
        
        
            
        


if __name__ == '__main__':
    train_simple_generative_model_on_mnist()
