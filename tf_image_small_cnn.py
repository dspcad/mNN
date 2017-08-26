#!/usr/bin/python

import cPickle
import numpy as np
import os
import sys
import tensorflow as tf
#import matplotlib.pyplot as plt

def unpickle(file):
  fo = open(file, 'rb')#open the file in binary mode
  dict = cPickle.load(fo)
  fo.close()
  return dict

def load_CIFAR10(folder):
  tr_data = np.empty((0,32*32*3), dtype=np.float32)
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
  batch_idx = np.random.randint(0,len(input_data),mini_batch)
  #batch_idx = xrange(start, start+mini_batch)

  #print len(input_data)
  #print len(batch_idx)

  img_batch = np.empty((32,32,3), dtype=np.float32)
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
  train_y = np.zeros((mini_batch,K), dtype=np.float32)
  for i in range(mini_batch):
    train_y[i][label_batch[i]] = 1


  #return img_batch, label_batch
  return img_batch, train_y


def batchTestRead(input_data, input_label):
  img_batch = np.empty((32,32,3), dtype=np.float32)
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
  test_y = np.zeros((len(input_data),K), dtype=np.float32)
  for i in range(len(input_data)):
    test_y[i][label_batch[i]] = 1


  return img_batch, test_y



if __name__ == '__main__':
  print '=========    Model Setting    ========='
  print "=  Model size: %s" % sys.argv[1]
  model_name = "./checkpoint/model_small_%s.ckpt" % sys.argv[2]
  print "=  Model name: %s " % model_name
  print '======================================='

  print '===== Start loadin CIFAR10 ====='
  datapath = '/home/hhwu/cifar-10-batches-py/'

  tr_data10, tr_labels10, te_data10, te_labels10, label_names10 = load_CIFAR10(datapath)
  print '  load CIFAR10 ... '

  print tr_data10.shape
  print tr_data10.dtype
  print tr_labels10.shape
  print te_data10.shape
  print te_labels10.dtype

  center_img = np.mean(tr_data10,axis=0)
  std_img = np.std(tr_data10,axis=0)

  print center_img
  print center_img.shape
  print std_img
  print std_img.shape
 
  tr_data10 = np.subtract(tr_data10, center_img, casting='unsafe')
  tr_data10 /= std_img

  #te_data10 = np.subtract(te_data10, center_img, casting='unsafe')
  #te_data10 /= std_img


  y = tr_labels10

  test_result = open("test_result.txt", 'w')

  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 100
  K = 10 # number of classes
  NUM_FILTER_1 = int(sys.argv[1])
  NUM_FILTER_2 = 32
#  NUM_FILTER_3 = 64
#  NUM_FILTER_4 = 64
#  NUM_FILTER_5 = 128
#  NUM_FILTER_6 = 128
#
#  NUM_NEURON_1 = 512
#  NUM_NEURON_2 = 512
#
  DROPOUT_PROB_1 = 1.00
  DROPOUT_PROB_2 = 1.00

  LEARNING_RATE = 1e-2
 
  reg = 5e-4 # regularization strength

  STDDEV = 5e-2

  # Dropout probability
  keep_prob_1 = tf.placeholder(tf.float32)
  keep_prob_2 = tf.placeholder(tf.float32)

  # initialize parameters randomly
  X  = tf.placeholder(tf.float32, shape=[None, 32,32,3])
  Y_ = tf.placeholder(tf.float32, shape=[None,K])

  W1 = tf.Variable(tf.truncated_normal([3,3,3,NUM_FILTER_1], stddev=STDDEV), name='w1')
  b1 = tf.Variable(tf.ones([NUM_FILTER_1])/10, name='b1')

  W2 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_1,NUM_FILTER_2], stddev=STDDEV), name='w2')
  b2 = tf.Variable(tf.ones([NUM_FILTER_2])/10, name='b2')

#  W3 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_2,NUM_FILTER_3], stddev=STDDEV))
#  b3 = tf.Variable(tf.ones([NUM_FILTER_3])/10)
#
#  W4 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_3,NUM_FILTER_4], stddev=STDDEV))
#  b4 = tf.Variable(tf.ones([NUM_FILTER_4])/10)
#
#  W5 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_4,NUM_FILTER_5], stddev=STDDEV))
#  b5 = tf.Variable(tf.ones([NUM_FILTER_5])/10)
#
#  W6 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_5,NUM_FILTER_6], stddev=STDDEV))
#  b6 = tf.Variable(tf.ones([NUM_FILTER_6])/10)
#
#
  W7 = tf.Variable(tf.truncated_normal([16*16*NUM_FILTER_2,K], stddev=STDDEV))
  b7 = tf.Variable(tf.ones([K])/10)
#
#  W8 = tf.Variable(tf.truncated_normal([NUM_NEURON_1,NUM_NEURON_2], stddev=STDDEV))
#  b8 = tf.Variable(tf.ones([NUM_NEURON_2])/10)

#  W9 = tf.Variable(tf.truncated_normal([NUM_FILTER_2,K], stddev=STDDEV))
#  b9 = tf.Variable(tf.ones([K])/10)

  #===== architecture =====#
  Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')+b1)
  Y2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,1,1,1], padding='SAME')+b2), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#  Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,1,1,1], padding='SAME')+b3)
#  Y4 = tf.nn.avg_pool(tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1,1,1,1], padding='SAME')+b4), ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
#
#  Y5 = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1,1,1,1], padding='SAME')+b5)
#  Y6 = tf.nn.avg_pool(tf.nn.relu(tf.nn.conv2d(Y5, W6, strides=[1,1,1,1], padding='SAME')+b6), ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


  YY = tf.reshape(Y2, shape=[-1,16*16*NUM_FILTER_2])

#  Y7 = tf.nn.relu(tf.matmul(YY,W7)+b7)
#  Y8 = tf.nn.relu(tf.matmul(Y7,W8)+b8)


  Y  = tf.nn.softmax(tf.matmul(YY,W7)+b7)


  #Y  = tf.nn.softmax(tf.matmul(YY,W7)+b7)

  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = LEARNING_RATE
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                            10000, 0.9, staircase=True)

  diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  cross_entropy = tf.reduce_mean(diff) + reg*sum(reg_losses)

  correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  # Passing global_step to minimize() will increment it at each step.
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
  train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=global_step)
 

  #####################################################
  #        Save the W1 and W2 parameters only         #
  #####################################################
  saver = tf.train.Saver([W1,W2]) 

  #learning_rate = tf.placeholder(tf.float32, shape=[])
  #train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Restore variables from disk.
  #saver.restore(sess, "./checkpoint/model_small.ckpt")
  #print("Model restored.")

  #te_x, te_y = batchTestRead(te_data10, te_labels10)
  #print '  Start training... '
  idx_start = 0
  #num_input_data =tr_data10.shape[0]
  for itr in xrange(10000):
    x, y = batchRead(tr_data10, tr_labels10, idx_start)
    sess.run(train_step, feed_dict={X: x, Y_: y, keep_prob_1: DROPOUT_PROB_1, keep_prob_2: DROPOUT_PROB_2})
 
    if itr % 100 == 0:
      print "iteration %d:  learning rate: %f  cross entropy: %f  accuracy: %f" % (itr,
                                                              #step_size,
                                                              learning_rate.eval(session=sess, feed_dict={X: x, Y_: y, 
                                                                                                          keep_prob_1: DROPOUT_PROB_1, 
                                                                                                          keep_prob_2: DROPOUT_PROB_2}),
                                                              cross_entropy.eval(session=sess, feed_dict={X: x, Y_: y, 
                                                                                                          keep_prob_1: DROPOUT_PROB_1, 
                                                                                                          keep_prob_2: DROPOUT_PROB_2}),
                                                              accuracy.eval(session=sess, feed_dict={X: x, Y_: y, 
                                                                                                     keep_prob_1: DROPOUT_PROB_1, 
                                                                                                     keep_prob_2: DROPOUT_PROB_2}))

    #  #save_path = saver.save(sess, "./checkpoint/model.ckpt")
    #  print("Model saved in file: %s" % save_path)
    #  print "Test Accuracy: %f" %  accuracy.eval(session=sess, feed_dict={X: te_x, Y_: te_y, keep_prob_1: DROPOUT_PROB_1, keep_prob_2: DROPOUT_PROB_2})
    #  test_result.write("Test Accuracy: %f" %  accuracy.eval(session=sess, feed_dict={X: te_x, Y_: te_y, 
    #                                                                                  keep_prob_1: DROPOUT_PROB_1, 
    #                                                                                  keep_prob_2: DROPOUT_PROB_2}))
    #  test_result.write("\n")

    #print "batch: ", idx_start
    if idx_start+mini_batch >= len(tr_data10):
      idx_start = 0
    else:
      idx_start += mini_batch



  #####################################################
  #                  Save the model                   #
  #####################################################
  save_path = saver.save(sess, model_name)



  #te_data10 = np.subtract(te_data10, center_img, casting='unsafe')
  #te_data10 = te_data10/std_img
  #te_x, te_y = batchTestRead(te_data10, te_labels10)
  #print "==================== Test Accuracy ===================="
  #print "Test Accuracy: %f" %  accuracy.eval(session=sess, feed_dict={X: te_x, Y_: te_y, keep_prob_1: DROPOUT_PROB_1,
  #                                                                                       keep_prob_2: DROPOUT_PROB_2})
  #print "=                                                     ="
  #print "======================================================="

  #test_result.write("Test Accuracy: %f" %  accuracy.eval(session=sess, feed_dict={X: te_x, Y_: te_y, 
  #                                                                                    keep_prob_1: DROPOUT_PROB_1, 
  #                                                                                    keep_prob_2: DROPOUT_PROB_2}))
  #test_result.write("\n")


  #x, y = batchTestRead(tr_data10, tr_labels10)
  #print "==================== Training Accuracy ===================="
  #print "Training Accuracy: %f" %  accuracy.eval(session=sess, feed_dict={X: x, Y_: y, keep_prob_1: DROPOUT_PROB_1,
  #                                                                                     keep_prob_2: DROPOUT_PROB_2})
  #print "=                                                     ="
  #print "==========================================================="
