#!/usr/bin/python

import cPickle
import numpy as np
import os
import tensorflow as tf
#import matplotlib.pyplot as plt

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



def batchRead(input_data, input_label,start):
  #batch_idx = np.random.randint(0,len(input_data),mini_batch)
  batch_idx = xrange(start, start+mini_batch)

  #print len(input_data)
  #print len(batch_idx)

  img_batch = np.empty((32,32,3))
  label_batch = np.empty(1)
  for i in range(0, mini_batch):
    #Convolutional layer
    img = input_data[batch_idx[i]].reshape((3,32,32)) #  1024  |  1024  | 1024 
                                           #    32  |    32  |  32   
                                           #    32  |    32  |  32   

    img = np.rollaxis(img, 0, 3) # 32 X 32 X 3

    if i == 0:
      img_batch = img
      label_batch = input_label[batch_idx[i]]
    else:
      img_batch = np.vstack((img_batch, img))
      label_batch = np.hstack((label_batch, input_label[batch_idx[i]]))
  
  img_batch = img_batch.reshape(mini_batch,32,32,3)

  #convert to one hot labels
  train_y = np.zeros((mini_batch,K))
  for i in range(mini_batch):
    train_y[i][label_batch[i]] = 1


  #return img_batch, label_batch
  return img_batch, train_y


def batchTestRead(input_data, input_label):
  img_batch = np.empty((32,32,3))
  label_batch = np.empty(1)
  for i in range(0, len(input_data)):
    #Convolutional layer
    img = input_data[i].reshape((3,32,32)) #  1024  |  1024  | 1024 
                                           #    32  |    32  |  32   
                                           #    32  |    32  |  32   

    img = np.rollaxis(img, 0, 3) # 32 X 32 X 3

    if i == 0:
      img_batch = img
      label_batch = input_label[i]
    else:
      img_batch = np.vstack((img_batch, img))
      label_batch = np.vstack((label_batch, input_label[i]))
  
  #print 'before', img_batch.shape
  img_batch = img_batch.reshape(len(input_data),32,32,3)
  #print 'after', img_batch.shape

  #convert to one hot labels
  test_y = np.zeros((len(input_data),K))
  for i in range(len(input_data)):
    test_y[i][label_batch[i]] = 1


  return img_batch, test_y




if __name__ == '__main__':
  print '===== Start loadin CIFAR10 ====='
  #datapath = '/home/hhwu/tensorflow_work/cifar-10-batches-py/'
  datapath = '/home/hhwu/tensorflow_work/cs231n/cifar-10-batches-py/'

  tr_data10, tr_labels10, te_data10, te_labels10, label_names10 = load_CIFAR10(datapath)
  print '  load CIFAR10 ... '

  print tr_data10.shape
  print tr_data10.dtype
  print tr_labels10.shape
  print te_data10.shape
  print te_labels10.dtype
 
  y = tr_labels10

  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 500
  K = 10 # number of classes
  NUM_FILTER_1 = 48
  NUM_FILTER_2 = 48

  NUM_FILTER_3 = 96
  NUM_FILTER_4 = 96

  NUM_FILTER_5 = 192
  NUM_FILTER_6 = 192


  NUM_NEURON_1 = 512
  NUM_NEURON_2 = 256

  reg = 5e-4 # regularization strength
  #step_size = 1
  step_size = 1e-3


  # initialize parameters randomly
  X  = tf.placeholder(tf.float32, shape=[None, 32,32,3])
  Y_ = tf.placeholder(tf.float32, shape=[None,K])

  W1 = tf.Variable(tf.truncated_normal([3,3,3,NUM_FILTER_1], stddev=0.1))
  b1 = tf.Variable(tf.ones([NUM_FILTER_1])/10)

  W2 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_1,NUM_FILTER_2], stddev=0.1))
  b2 = tf.Variable(tf.ones([NUM_FILTER_2])/10)

  W3 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_2,NUM_FILTER_3], stddev=0.1))
  b3 = tf.Variable(tf.ones([NUM_FILTER_3])/10)

  W4 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_3,NUM_FILTER_4], stddev=0.1))
  b4 = tf.Variable(tf.ones([NUM_FILTER_4])/10)

  W5 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_4,NUM_FILTER_5], stddev=0.1))
  b5 = tf.Variable(tf.ones([NUM_FILTER_5])/10)

  W6 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_5,NUM_FILTER_6], stddev=0.1))
  b6 = tf.Variable(tf.ones([NUM_FILTER_6])/10)


  W7 = tf.Variable(tf.truncated_normal([4*4*NUM_FILTER_6,NUM_NEURON_1], stddev=0.1))
  b7 = tf.Variable(tf.ones([NUM_NEURON_1])/10)

  W8 = tf.Variable(tf.truncated_normal([NUM_NEURON_1,NUM_NEURON_2], stddev=0.1))
  b8 = tf.Variable(tf.ones([NUM_NEURON_2])/10)

  W9 = tf.Variable(tf.truncated_normal([NUM_NEURON_2,K], stddev=0.1))
  b9 = tf.Variable(tf.ones([K])/10)

  Y1  = tf.nn.relu(tf.nn.conv2d(X,  W1, strides=[1,1,1,1], padding='SAME')+b1)
  Y2  = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,1,1,1], padding='SAME')+b2)
  YP1 = tf.nn.max_pool(Y2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  
  Y3  = tf.nn.relu(tf.nn.conv2d(YP1, W3, strides=[1,1,1,1], padding='SAME')+b3)
  Y4  = tf.nn.relu(tf.nn.conv2d(Y3,  W4, strides=[1,1,1,1], padding='SAME')+b4)
  YP2 = tf.nn.max_pool(Y4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  Y5  = tf.nn.relu(tf.nn.conv2d(YP2, W5, strides=[1,1,1,1], padding='SAME')+b5)
  Y6  = tf.nn.relu(tf.nn.conv2d(Y5,  W6, strides=[1,1,1,1], padding='SAME')+b6)
  YP3 = tf.nn.max_pool(Y6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

 
  YY = tf.reshape(YP3, shape=[-1,4*4*NUM_FILTER_6])

  Y7 = tf.nn.relu(tf.matmul(YY,W7)+b7)
  Y8 = tf.nn.relu(tf.matmul(Y7,W8)+b8)
  Y  = tf.nn.softmax(tf.matmul(Y8,W9)+b9)


  diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  cross_entropy = tf.reduce_mean(diff) + reg*sum(reg_losses)

  correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  learning_rate = tf.placeholder(tf.float32, shape=[])
  train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  idx_start = 0
  epoch = 0
  #num_input_data =tr_data10.shape[0]
  for itr in xrange(100000):
    x, y = batchRead(tr_data10, tr_labels10, idx_start)
    sess.run(train_step, feed_dict={X: x, Y_: y, learning_rate: step_size})
 
    if itr % 10 == 0:
      print "iteration %d:  learning rate: %f  cross entropy: %f  accuracy: %f" % (itr,
                                                              step_size,
                                                              cross_entropy.eval(session=sess, feed_dict={X: x, Y_: y}),
                                                              accuracy.eval(session=sess, feed_dict={X: x, Y_: y}))
    #print "batch: ", idx_start
    if idx_start+mini_batch >= len(tr_data10):
      idx_start = 0
      epoch += 1
    else:
      idx_start += mini_batch

    if epoch == 20:
      step_size = step_size/2
      epoch = 0


  x, y = batchTestRead(te_data10, te_labels10)
  print(accuracy.eval(session=sess, feed_dict={X: x, Y_: y}))
