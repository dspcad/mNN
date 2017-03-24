#!/usr/bin/python

import cPickle
import numpy as np
import os
import matplotlib.pyplot as plt

import pprint

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
            tr_data = data_dict['data'].reshape(10000, 32, 32, 3)
            print tr_data.shape
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
 
  X = tr_data10.T
  y = tr_labels10
  num_examples = X.shape[1]
  y = np.reshape(tr_labels10, (1,tr_data10.shape[0]))
  print y.shape

  N = 5000 # number of points per class
  D = 3072 # dimensionality
  K = 10 # number of classes

  X_test = te_data10.T
  y_test = np.reshape(te_labels10, (1,te_data10.shape[0]))

#===== Softmax linear classifier =====

  
  #Train a Linear Classifier
  W = 0.01 * np.random.randn(K,D)/np.sqrt(num_examples)
  b = np.zeros((K,1))

  # some hyperparameters
  step_size = 2e-7
  reg = 1e-3 # regularization strength

  # gradient descent loop
  for i in xrange(50):
  
    # evaluate class scores, [K x (N*K)]
    scores = np.dot(W,X) + b
    max_scores = np.amax(scores, axis=0)
    scores = np.subtract(scores, max_scores)

    # compute the class probabilities
    exp_scores = np.exp(scores)
    #print "exp scores"
    #print exp_scores
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    #print "prob scores"
    #print probs

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs.T[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
      predicted_class = np.argmax(scores, axis=0)
      print "iteration %d: loss %f training accuracy: %.2f" % (i, loss, np.mean(predicted_class == y))


    # compute the gradient on scores
    dscores = probs.T #[(N*K) X K]
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
    
    
    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(dscores.T, X.T)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg*W # regularization gradient
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db.T
  

  # evaluate training set accuracy
  scores = np.dot(W, X) + b
  predicted_class = np.argmax(scores, axis=0)
  print 'training accuracy: %.2f' % (np.mean(predicted_class == y))

#===== two-layer NN =====

  # initialize parameters randomly
  h = 50 # size of hidden layer
  W1 = 0.01 * np.random.randn(h,D)/np.sqrt(num_examples)
  b1 = np.zeros((h,1))
  W2 = 0.01 * np.random.randn(K,h)/np.sqrt(num_examples)
  b2 = np.zeros((K,1))

  # some hyperparameters
  step_size = 5e-4
  reg = 1e-3 # regularization strength

  # gradient descent loop
  num_examples = X.shape[1]
  for i in xrange(1000):
 
    # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(W1,X) + b1)
    scores = np.dot(W2,hidden_layer) + b2

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs.T[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if i % 10 == 0:
      predicted_class = np.argmax(scores, axis=0)
      print "iteration %d: loss %f training accuracy: %.2f" % (i, loss, np.mean(predicted_class == y))


    # compute the gradient on scores
    dscores = probs.T #[N X K]
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters (W,b)
    # first backprop into parameters W2 and b2
    dW2 = np.dot(dscores.T, hidden_layer.T)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer.T <= 0] = 0
    # finally into W,b
    dW1 = np.dot(dhidden.T, X.T)
    db1 = np.sum(dhidden, axis=0, keepdims=True)
 
    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg * W1
 
    # perform a parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1.T
    W2 += -step_size * dW2
    b2 += -step_size * db2.T
  

  # evaluate training set accuracy
  hidden_layer = np.maximum(0, np.dot(W1,X) + b1)
  scores = np.dot(W2,hidden_layer) + b2
  predicted_class = np.argmax(scores, axis=0)
  print 'training accuracy: %.2f' % (np.mean(predicted_class == y))


  # evaluate test set accuracy
  hidden_layer = np.maximum(0, np.dot(W1,X_test) + b1)
  scores = np.dot(W2,hidden_layer) + b2
  predicted_class = np.argmax(scores, axis=0)
  print 'test accuracy: %.2f' % (np.mean(predicted_class == y_test))


