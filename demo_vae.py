from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import os

def vae_train(sess, optimizer, cost, x, n_samples, batch_size=100, learn_rate=0.001, train_epoch=10, verb=1, verb_step=5):
    """
    Defined for MNIST Dataset
    """
    for epoch in range(train_epoch):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_x, _ = mnist.train.next_batch(batch_size)
            
            _, c = sess.run((optimizer, cost), feed_dict={x: batch_x})
            avg_cost += c / n_samples * batch_size
        
        if verb:
            if epoch % verb_step == 0:
                print('Epoch:%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))

# Prepare Dataset
from misc_sta663 import *
(mnist, n_samples) = mnist_loader()

# Normal Tensorflow
from vae_parallel_sta663 import *
import tensorflow as tf
import numpy as np

config_normal = {}
config_normal['x_in'] = 784
config_normal['encoder_1'] = 500
config_normal['encoder_2'] = 500
config_normal['decoder_1'] = 500
config_normal['decoder_2'] = 500
config_normal['z'] = 20

print('For demonstrating purpose, we limited the epoch number to 50')
(sess, optimizer, cost, x, x_prime) = vae_init_parallel()
print("Begin training, this might take some time...")
vae_train(sess, optimizer, cost, x, n_samples, train_epoch=51)

examples_to_show = 100
x_sample = mnist.test.images[:examples_to_show]
x_reconstruct = sess.run(x_prime,  feed_dict={x: x_sample})

plt.figure(figsize=(8, 2))
for i in range(8):

    ax = plt.subplot(2, 8, i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    
    ax = plt.subplot(2, 8, 8 + i + 1)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    
plt.tight_layout()
plt.show()