import numpy as np
import tensorflow as tf

# number of device count
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_cpu_core', 1, 'Number of CPU cores to use')
tf.app.flags.DEFINE_integer('intra_op_parallelism_threads', 1, 'How many ops can be launched in parallel')
tf.app.flags.DEFINE_integer('num_gpu_core', 0, 'Number of GPU cores to use')
device_id = -1

def next_device(use_cpu = True):
    ''' See if there is available next device;
        Args: use_cpu, global device_id
        Return: new device id
    '''
    global device_id
    if (use_cpu):
        if ((device_id + 1) < FLAGS.num_cpu_core):
            device_id += 1
        device = '/cpu:%d' % device_id
    else:
        if ((device_id + 1) < FLAGS.num_gpu_core):
            device_id += 1
        device = '/gpu:%d' % device_id
    return device

def xavier_init(neuron_in, neuron_out, constant=1):
    low = -constant*np.sqrt(6/(neuron_in + neuron_out))
    high = constant*np.sqrt(6/(neuron_in + neuron_out))
    return tf.random_uniform((neuron_in, neuron_out), minval=low, maxval=high, dtype=tf.float32)

def init_weights(config):
    """
    Initialize weights with specified configuration using Xavier algorithm
    """
    encoder_weights = dict()
    decoder_weights = dict()
    
    # two layers encoder
    encoder_weights['h1'] = tf.Variable(xavier_init(config['x_in'], config['encoder_1']))
    encoder_weights['h2'] = tf.Variable(xavier_init(config['encoder_1'], config['encoder_2']))
    encoder_weights['mu'] = tf.Variable(xavier_init(config['encoder_2'], config['z']))
    encoder_weights['sigma'] = tf.Variable(xavier_init(config['encoder_2'], config['z']))
    encoder_weights['b1'] = tf.Variable(tf.zeros([config['encoder_1']], dtype=tf.float32))
    encoder_weights['b2'] = tf.Variable(tf.zeros([config['encoder_2']], dtype=tf.float32))
    encoder_weights['bias_mu'] = tf.Variable(tf.zeros([config['z']], dtype=tf.float32))
    encoder_weights['bias_sigma'] = tf.Variable(tf.zeros([config['z']], dtype=tf.float32))
    
    # two layers decoder
    decoder_weights['h1'] = tf.Variable(xavier_init(config['z'], config['decoder_1']))
    decoder_weights['h2'] = tf.Variable(xavier_init(config['decoder_1'], config['decoder_2']))
    decoder_weights['mu'] = tf.Variable(xavier_init(config['decoder_2'], config['x_in']))
    decoder_weights['sigma'] = tf.Variable(xavier_init(config['decoder_2'], config['x_in']))
    decoder_weights['b1'] = tf.Variable(tf.zeros([config['decoder_1']], dtype=tf.float32))
    decoder_weights['b2'] = tf.Variable(tf.zeros([config['decoder_2']], dtype=tf.float32))
    decoder_weights['bias_mu'] = tf.Variable(tf.zeros([config['x_in']], dtype=tf.float32))
    decoder_weights['bias_sigma'] = tf.Variable(tf.zeros([config['x_in']], dtype=tf.float32))
    
    return (encoder_weights, decoder_weights)


def forward_z(x, encoder_weights):
    """
    Compute mean and sigma of z
    """
    with tf.device(next_device()):
        layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, encoder_weights['h1']), encoder_weights['b1']))
    with tf.device(next_device()):
        layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, encoder_weights['h2']), encoder_weights['b2']))
    z_mean = tf.add(tf.matmul(layer_2, encoder_weights['mu']), encoder_weights['bias_mu'])
    z_sigma = tf.add(tf.matmul(layer_2, encoder_weights['sigma']), encoder_weights['bias_sigma'])
    
    return(z_mean, z_sigma)


def reconstruct_x(z, decoder_weights):
    """
    Use z to reconstruct x
    """
    with tf.device(next_device()):
        layer_1 = tf.nn.softplus(tf.add(tf.matmul(z, decoder_weights['h1']), decoder_weights['b1']))
    with tf.device(next_device()):
        layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, decoder_weights['h2']), decoder_weights['b2']))
    x_prime = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, decoder_weights['mu']), decoder_weights['bias_mu']))
    
    return x_prime


def optimize_func(z, z_mean, z_sigma, x, x_prime, learn_rate):
    """
    Define cost and optimize function
    """
    # define loss function
    # reconstruction lost
    recons_loss = -tf.reduce_sum(x * tf.log(1e-10 + x_prime) + (1-x) * tf.log(1e-10 + 1 - x_prime), 1)
    # KL distance
    latent_loss = -0.5 * tf.reduce_sum(1 + z_sigma - tf.square(z_mean) - tf.exp(z), 1)
    # summing two loss terms together
    cost = tf.reduce_mean(recons_loss + latent_loss)
    
    # use ADAM to optimize
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
    
    return (cost, optimizer)

def vae_init_parallel(batch_size=100, learn_rate=0.001, config={}):
    """
    This function build a varational autoencoder based on https://jmetzen.github.io/2015-11-27/vae.html
    In consideration of simplicity and future work on optimization, we removed the class structure
    A tensorflow session, optimizer and cost function as well as input data will be returned
    """
    # default configuration of network
    # x_in = 784
    # encoder_1 = 500
    # encoder_2 = 500
    # decoder_1 = 500
    # decoder_2 = 500
    # z = 20
    
    # use default setting if no configuration is specified
    if not config:
        config['x_in'] = 784
        config['encoder_1'] = 500
        config['encoder_2'] = 500
        config['decoder_1'] = 500
        config['decoder_2'] = 500
        config['z'] = 20
    
    # input
    x = tf.placeholder(tf.float32, [None, config['x_in']])
    
    # initialize weights
    (encoder_weights, decoder_weights) = init_weights(config)
    
    # compute mean and sigma of z
    (z_mean, z_sigma) = forward_z(x, encoder_weights)
    
    # compute z by drawing sample from normal distribution
    eps = tf.random_normal((batch_size, config['z']), 0, 1, dtype=tf.float32)
    z_val = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_sigma)), eps))
    
    # use z to reconstruct the network
    x_prime = reconstruct_x(z_val, decoder_weights)
    
    # define loss function
    (cost, optimizer) = optimize_func(z_val, z_mean, z_sigma, x, x_prime, learn_rate)
    
    # initialize all variables
    init = tf.global_variables_initializer()
    
    # parallel configuration
    config_ = tf.ConfigProto(device_count={"CPU": FLAGS.num_cpu_core}, # limit to num_cpu_core CPU usage  
                             #inter_op_parallelism_threads = 1,   
                             #intra_op_parallelism_threads = FLAGS.intra_op_parallelism_threads,  
                             #log_device_placement=True
                            )  
    
    # define and return the session
    sess = tf.InteractiveSession(config=config_)
    sess.run(init)
    
    return (sess, optimizer, cost, x, x_prime)