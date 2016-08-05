# %% Imports
import tensorflow as tf
import numpy as np
import math

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

import tool as tl
datas_train, datas_test, labels_train, labels_test = tl.getMaterial()

from liblinearutil import *
import statistics
def tune(epochs=[200,500], nodes=[64,256], s_range=6, c_from=1.5, c_to=10.5, c_step=0.5, *d_bound):
    training_epochs = epochs
    hidden_nodes = nodes
    costmin_list = [] 
    epoch_best_list = []
    node_best_list = []
    accmax_list = []
    s_best_list = []
    c_best_list = []
    feat_best_list = list()
    feat_test_list = list()
    d_range = range(len(datas_train))
    if len(d_bound[0]) == 2:
        d_range = range(d_bound[0][0]-1, d_bound[0][1])
    else:
        #raise TypeError("Wrong data boundary assignment!")
        print(len(d_bound))
    for data_i in d_range:
        costmin = 1
        epoch_best = 0
        node_best = 0
        accmax = 0
        s_best = 0
        c_best = 0
        feat_best = []
        feat_best_t = []
        for node_num in hidden_nodes:
            sess = tf.Session() 
            for epoch_num in training_epochs:
                ae = autoencoder(dimensions=[310, node_num])
                optimizer = tf.train.AdamOptimizer(0.001).minimize(ae['cost'])
                sess.run(tf.initialize_all_variables())

                for i in range(epoch_num):
                    sess.run(optimizer, feed_dict={ae['x']: datas_train[data_i]}) 
                    
                feat_train = sess.run(ae['z'], feed_dict={ae['x']: datas_train[data_i]})    
                feat_test, cost = sess.run([ae['z'], ae['cost']], feed_dict={ae['x']: datas_test[data_i]})                
                #data_train_list = datas_train[data_i].tolist()
                feat_train_list = feat_train.tolist()
                feat_test_list = feat_test.tolist()
                label_train_list = labels_train[data_i].T[0].tolist()
                label_test_list = labels_test[data_i].T[0].tolist()

                prob = problem(label_train_list, feat_train_list)
                for s in range(s_range):
                    for c in np.arange(c_from, c_to, c_step):
                        #print(s,c)
                        param = parameter('-s %d -c %f -q'%(s, c))
                        m = train(prob, param)
                        p_labels, p_acc, p_vals = predict(label_test_list, feat_test_list, m)
                        if p_acc[0] > accmax:
                            accmax = p_acc[0]
                            s_best = s
                            c_best = c
                            costmin = cost           
                            epoch_best = epoch_num
                            node_best = node_num
                            feat_best = feat_train_list
                            feat_best_t = feat_test
        accmax_list.append(accmax)
        s_best_list.append(s_best)
        c_best_list.append(c_best)
        costmin_list.append(costmin)
        epoch_best_list.append(epoch_best)
        node_best_list.append(node_best)
        feat_best_list.append(feat_best)
        feat_test_list.append(feat_best_t)
        print("Complete %d/%d!"%(data_i+1, d_bound[0][1]-d_bound[0][0]+1))

    print(accmax_list)
    print(sum(accmax_list)/len(accmax_list))
    print(s_best_list)
    print(c_best_list)
    
    #s_most = statistics.mode(s_best_list)
    #c_most = statistics.mode(c_best_list)
    #print(s_most, c_most)
    #label_train = [labels_train[i] for i in d_range] 
    #label_test = [labels_test[i] for i in d_range]
    #print(np.shape(label_train), np.shape(feat_best_list))
    #
    #param = parameter('-s %d -c %f -q'%(s_most, c_most))
    #m = train(prob, param)
    #p_labels, p_acc, p_vals = predict(label_test, feat_test_list, m)
tune([200],[64],6,1.5,10.5,5,[1,3])
