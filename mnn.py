#!/usr/bin/python

import cPickle
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
  N = 100 # number of points per class
  D = 2 # dimensionality
  K = 3 # number of classes
  XT = np.zeros((N*K,D)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels
  for j in xrange(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    XT[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
  # lets visualize the data:
  plt.scatter(XT[:, 0], XT[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  X = XT.T
  #plt.show()
  
  #Train a Linear Classifier
  W = 0.01 * np.random.randn(K,D)
  b = np.zeros((K,1))

  # some hyperparameters
  step_size = 1e-0
  reg = 1e-3 # regularization strength

  # gradient descent loop
  num_examples = X.shape[1]
  for i in xrange(200):
  
    # evaluate class scores, [K x (N*K)]
    scores = np.dot(W,X) + b
    #print scores
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs.T[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
      print "iteration %d: loss %f" % (i, loss)


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
  h = 100 # size of hidden layer
  W1 = 0.01 * np.random.randn(h,D)
  b1 = np.zeros((h,1))
  W2 = 0.01 * np.random.randn(K,h)
  b2 = np.zeros((K,1))

  # some hyperparameters
  step_size = 1e-0
  reg = 1e-3 # regularization strength

  # gradient descent loop
  num_examples = X.shape[1]
  for i in xrange(10000):
 
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
    if i % 1000 == 0:
      print "iteration %d: loss %f" % (i, loss)


    # compute the gradient on scores
    dscores = probs.T #[(N*K) X K]
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


