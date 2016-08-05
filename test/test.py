# %% Imports
import tensorflow as tf
import numpy as np
import math

def testoption(a, *c):
    if c:
        raise TypeError("YA")
        print("Only c!")
    else:
        print("Only a!")

# %% Autoencoder definition
def autoencoder(dimensions=[310, 256, 128]):
    # %% input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')

    # %% Build the encoder
    encoder = []
    encoder_input = x

    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(encoder_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(encoder_input, W) + b)
        encoder_input = output

    # %% latent representation
    z = encoder_input
    encoder.reverse()
    
    decoder_input = z
    # %% Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(decoder_input, W) + b)
        decoder_input = output

    # %% now have the reconstruction through the network
    y = decoder_input

    # %% cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x))
    return {'x': x, 'z': z, 'y': y, 'cost': cost}

testoption(5,10)


#import tool as tl
#datas_train, datas_test, labels_train, labels_test = tl.getMaterial()
#sess = tf.Session()
#ae = autoencoder([310, 64])
#optimizer = tf.train.AdamOptimizer(0.001).minimize(ae['cost'])
#sess.run(tf.initialize_all_variables())
#
#for i in range(200):
#    sess.run(optimizer, feed_dict={ae['x']: datas_train[0]})
#    print(sess.run(ae['cost'], feed_dict={ae['x']: datas_train[0]}))
