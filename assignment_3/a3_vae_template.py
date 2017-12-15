import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm


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


class VariationalAutoencoder(object):
    
    def __init__(self,batch_size,z_dim,kernel_init,activation,encoder_dims,decoder_dims):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.kernel_init = tf.keras.initializers.glorot_uniform()#kernel_init
        self.activation = tf.nn.relu
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

    def lower_bound(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of the lower-bound on the log-probability of each data point
        """
        z_mean,z_log_var = self.encoder(x)
        z = self.reparam((z_mean,z_log_var))#tf.keras.layers.Lambda(self.reparam, output_shape=(self.z_dim,))([z_mean, z_log_var])
        x_recon = self.decoder(z)
        
        kl_loss = -.5 * tf.reduce_sum(1. + z_log_var - tf.square(z_mean) - 
                                      tf.exp(z_log_var), axis=-1)
        xent_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = x_recon, labels = x),
                                  axis=-1)#       
        return -tf.reduce_mean(xent_loss + kl_loss)
    

    def mean_x_given_z(self, z):
        """
        :param z: A (n_samples, n_dim_z) tensor containing a set of latent data points (n_samples, n_dim_z)
        :return: A (n_samples, n_dim_x) tensor containing the mean of p(X|Z=z) for each of the given points
        """
        pass

    def sample(self, n_samples):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """
        z = tf.random_normal(shape=[n_samples,self.z_dim])        
        x_decoded = self.decoder(z)
        probs = tf.distributions.Bernoulli(logits=x_decoded)
        
        return probs.sample()
    
    def linear(self,ipt, output_dim, scope=None, stddev=1.0):
        with tf.variable_scope(scope or 'linear',reuse=tf.AUTO_REUSE):
            w = tf.get_variable(
                'w',
                [ipt.get_shape()[1], output_dim],
                initializer=self.kernel_init,
            )
            b = tf.get_variable(
                'b',
                [output_dim],
                initializer=self.kernel_init
            )
            return tf.matmul(ipt, w) + b
        
    def encoder(self,x):
        h_in = x
        for j,i in enumerate(self.encoder_dims):
            scope = 'n'+str(j)
            h_in = self.activation(self.linear(h_in,i,scope))
            
        z_mean = self.linear(h_in,self.z_dim,"zmean")
        z_log_var = self.linear(h_in,self.z_dim,"zlogvar")
        return z_mean,z_log_var        
        
    def reparam(self,args):
        z_mean, z_log_var = args
        epsilon = tf.random_normal(tf.shape(z_log_var))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon       
        
    def decoder(self,z):
        h_in = z
        for j,i in enumerate(self.decoder_dims):
            scope = 'm'+str(j)
            h_in = self.activation(self.linear(h_in,i,scope))
        x_recon = self.linear(h_in,784,"recon")
        return x_recon
        


def train_vae_on_mnist(z_dim=2, kernel_initializer='glorot_uniform', optimizer = 'adam',  learning_rate=0.001, n_epochs=4000,
        test_every=100, minibatch_size=100, encoder_hidden_sizes=[200, 200], decoder_hidden_sizes=[200, 200],
        hidden_activation='relu', plot_grid_size=10, plot_n_samples = 20):
    """
    Train a variational autoencoder on MNIST and plot the results.

    :param z_dim: The dimensionality of the latent space.
    :param kernel_initializer: How to initialize the weight matrices (see tf.keras.layers.Dense)
    :param optimizer: The optimizer to use
    :param learning_rate: The learning rate for the optimizer
    :param n_epochs: Number of epochs to train
    :param test_every: Test every X training iterations
    :param minibatch_size: Number of samples per minibatch
    :param encoder_hidden_sizes: Sizes of hidden layers in encoder
    :param decoder_hidden_sizes: Sizes of hidden layers in decoder
    :param hidden_activation: Activation to use for hidden layers of encoder/decoder.
    :param plot_grid_size: Number of rows, columns to use to make grid-plot of images corresponding to latent Z-points
    :param plot_n_samples: Number of samples to draw when plotting samples from model.
    """

    # Get Data
    summary_path = "./summaries"
    
    x_train, x_test = load_mnist_images(binarize=True)
    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors
    
    test_iterator = tf.data.Dataset.from_tensor_slices(x_test).repeat().batch(minibatch_size).make_initializable_iterator()
    t_minibatch = test_iterator.get_next()

    # Build Model
    model = VariationalAutoencoder(minibatch_size,z_dim,kernel_initializer,
                                   hidden_activation,encoder_hidden_sizes,decoder_hidden_sizes)
    loss = model.lower_bound(x_minibatch)
    loss_t = model.lower_bound(t_minibatch)
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss)    

    with tf.Session() as sess:       
        sess.run(train_iterator.initializer)  # Initialize the variables of the data-loader.
        sess.run(test_iterator.initializer)
        sess.run(tf.global_variables_initializer())  # Initialize the model parameters.
        
        summary_f = tf.summary.FileWriter(summary_path,sess.graph)
        s_loss_train = tf.summary.scalar('train_loss',loss)
        s_loss_test = tf.summary.scalar('test_loss',loss_t)
    
        n_steps = (n_epochs * n_samples)/minibatch_size
        nn_samples = 16
        for i in range(int(n_steps)):
            opt,ls,sltr = sess.run([optim,loss,s_loss_train])
            
            summary_f.add_summary(sltr,i)
            
            if i % test_every==0:
                test_loss,slts = sess.run([loss_t,s_loss_test])
                summary_f.add_summary(slts,i)
                print("Train Step {:04d}/{:04d}, Batch Size = {}, Train Loss = {}, Test Loss = {}"
                      .format(i + 1,int(n_steps), minibatch_size, ls, test_loss))                
                
            
            if i% 1000==0:
                samples = model.sample(nn_samples)
                print(samples.shape)
                for j in range(nn_samples):
                    plt.subplot(4, 4, j + 1)
                    #plt.text(0, 1, samples[j], color='black', backgroundcolor='white', fontsize=8)
                    plt.imshow(tf.reshape(samples[j], shape=[28,28]).eval(), cmap='hot')
                    plt.axis('off')
    
                plt.savefig('VAE_%s.png' % str(i))
                plt.close()
                
            if i% 10000==0:
                nx = ny = 15
                x_values = np.linspace(.05, .95, nx)
                y_values = np.linspace(.05, .95, ny)                 
                canvas = np.empty((28*ny, 28*nx))
                for i, yi in enumerate(x_values):
                    for j, xi in enumerate(y_values):
                        z_mu = np.array([[norm.ppf(xi), norm.ppf(yi)]]).astype('float32')
                        z_mu = tf.convert_to_tensor(z_mu,dtype=tf.float32)
                        x_mean = model.decoder(z_mu)
                        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = tf.reshape(x_mean[0],shape=[28,28]).eval()
                        #x_mean[0].reshape(28, 28)
                 
                plt.figure(figsize=(10, 10))
                plt.imshow(canvas, origin="upper", cmap="gray")
                plt.tight_layout()
                plt.savefig('Manifold_%s.png' % str(i))
                plt.close()


if __name__ == '__main__':
    train_vae_on_mnist()
