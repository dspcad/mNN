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


#===== Softmax linear classifier =====

  #sess = tf.InteractiveSession()
  #Train a Linear Classifier

  with tf.name_scope('input'):
    x  = tf.placeholder(tf.float32, shape=[None, D], name='x-input')
  #W = 0.01*tf.Variable(tf.random_normal([D,K]))/np.sqrt(num_examples)

  with tf.name_scope('weights'):
    W = tf.Variable(0.01 * tf.random_normal([D,K])/num_examples)
    variable_summaries(W)

  with tf.name_scope('biases'):
    b = tf.Variable(tf.zeros([K]))
    variable_summaries(b)

  with tf.name_scope('Wx_plus_b'):
    y = tf.matmul(x,W) + b
    tf.summary.histogram('pre_activations', y)

  y_ = tf.placeholder(tf.float32, shape=[None,K])

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      cross_entropy = tf.reduce_mean(diff) + 0.0005*sum(reg_losses)

  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(2e-7).minimize(cross_entropy)


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
  for i in range(3000):
    summary, _ = sess.run([merged, train_step], feed_dict={x: tr_data10, y_: train_y})

    if i % 10 == 0:
      train_writer.add_summary(summary, i)
      print "iteration %d: cross entropy: %f accuracy: %f" % (i,
                                                              cross_entropy.eval(session=sess, feed_dict={x: tr_data10, y_: train_y}), 
                                                              accuracy.eval(session=sess, feed_dict={x: tr_data10, y_: train_y}))

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(accuracy.eval(session=sess, feed_dict={x: te_data10, y_: test_y}))


