#!/usr/bin/python

import cPickle
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

def unpickle(file):
  fo = open(file, 'rb')#open the file in binary mode
  dict = cPickle.load(fo)
  fo.close()
  return dict

def load_CIFAR10(folder):
    tr_data = np.empty((0,32*32*3))
    tr_labels = np.empty(1)
    '''
    32x32x3
    '''
    for i in range(1,6):
        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            tr_data = data_dict['data']
            tr_labels = data_dict['labels']
        else:
            tr_data = np.vstack((tr_data, data_dict['data']))
            tr_labels = np.hstack((tr_labels, data_dict['labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    te_data = data_dict['data']
    te_labels = np.array(data_dict['labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    label_names = bm['label_names']

    return tr_data, tr_labels, te_data, te_labels, label_names


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


if __name__ == '__main__':
  print '===== Start loadin CIFAR10 ====='
  datapath = '/home/hhwu/tensorflow_work/cs231n/cifar-10-batches-py/'
  tr_data10, tr_labels10, te_data10, te_labels10, label_names10 = load_CIFAR10(datapath)
  print '  load CIFAR10 ... '

  print tr_data10.shape
  print tr_data10.dtype
  print tr_labels10.shape
  print te_data10.shape
  print te_labels10.dtype

  N = 5000 # number of points per class
  D = 3072 # dimensionality
  K = 10 # number of classes
  h = 100 # size of hidden layer
  
  step_size = 5e-4

  #X = tr_data10.T
  #y = tr_labels10
  num_examples = te_data10.shape[0]
  test_y = np.zeros((num_examples,K))
  for i in range(num_examples):
    test_y[i][te_labels10[i]] = 1

  num_examples = tr_data10.shape[0]
  train_y = np.zeros((num_examples,K))
  for i in range(num_examples):
    train_y[i][tr_labels10[i]] = 1

  #y = np.reshape(tr_labels10, (1,tr_data10.shape[0]))
  #print y.shape


#===== two-layer NN =====

  # initialize parameters randomly

  with tf.name_scope('input'):
    x  = tf.placeholder(tf.float32, shape=[None, D], name='x-input')

  with tf.name_scope('weights'):
    with tf.name_scope('wights_1'):
      W1 = tf.Variable(0.01 * tf.random_normal([D,h])/num_examples)
      variable_summaries(W1)
    with tf.name_scope('wights_2'):
      W2 = tf.Variable(0.01 * tf.random_normal([h,K])/num_examples)
      variable_summaries(W2)

  with tf.name_scope('biases'):
    with tf.name_scope('biases_1'):
      b1 = tf.Variable(tf.zeros([h]))
      variable_summaries(b1)
    with tf.name_scope('biases_2'):
      b2 = tf.Variable(tf.zeros([K]))
      variable_summaries(b2)


  with tf.name_scope('W1x_plus_b1'):
    layer_1 = tf.matmul(x,W1) + b1
    hidden_layer = tf.nn.relu(layer_1)
    tf.summary.histogram('pre_activations', hidden_layer)

  with tf.name_scope('W2h_plus_b2'):
    y = tf.matmul(hidden_layer,W2) + b2
    tf.summary.histogram('pre_activations', y)

  y_ = tf.placeholder(tf.float32, shape=[None,K])

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      cross_entropy = tf.reduce_mean(diff) + 0.0005*sum(reg_losses)

  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    #train_step = tf.train.GradientDescentOptimizer(1e-7).minimize(cross_entropy)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)


  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('logs' + '/train', sess.graph)
  # Train
  for i in xrange(1000):
    summary, _ = sess.run([merged, train_step], feed_dict={x: tr_data10, y_: train_y, learning_rate: step_size})

    if i % 10 == 0:
      train_writer.add_summary(summary, i)
      print "iteration %d:  learning rate: %f  cross entropy: %f  accuracy: %f" % (i,
                                                              step_size,
                                                              cross_entropy.eval(session=sess, feed_dict={x: tr_data10, y_: train_y}), 
                                                              accuracy.eval(session=sess, feed_dict={x: tr_data10, y_: train_y}))


    if i == 10000:
      step_size = 1e-4


    if i == 20000:
      step_size = 5e-5

    if i == 30000:
      step_size = 1e-5


  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(accuracy.eval(session=sess, feed_dict={x: te_data10, y_: test_y}))




