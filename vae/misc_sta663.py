def mnist_loader():
    """
    Load MNIST data in tensorflow readable format
    The script comes from:
    https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
    """
    import gzip
    import os
    import tempfile
    import numpy
    from six.moves import urllib
    from six.moves import xrange
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    
    mnist = read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples
    
    return (mnist, n_samples)