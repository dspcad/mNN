#!/usr/bin/python

import cPickle
import numpy as np
import os
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


#def im2col(img, W, F, S, P):
def im2col(img):
  output_size = (W-F+2*P)/S + 1
  filter_data = np.empty([F*F*3, output_size*output_size])
  extend_image_size = 32+2*P
  extend_image = np.zeros([extend_image_size,extend_image_size,3], dtype=int)
  
  for i in range(0, extend_image_size):
    for k in range (0, P):
      extend_image[k][i][0]  = 0
      extend_image[k][i][1]  = 0
      extend_image[k][i][2]  = 0

      # 35, 34
      extend_image[extend_image_size-1-k][i][0] = 0
      extend_image[extend_image_size-1-k][i][1] = 0
      extend_image[extend_image_size-1-k][i][2] = 0

      extend_image[i][k][0]  = 0
      extend_image[i][k][1]  = 0
      extend_image[i][k][2]  = 0

      extend_image[i][extend_image_size-1-k][0] = 0
      extend_image[i][extend_image_size-1-k][1] = 0
      extend_image[i][extend_image_size-1-k][2] = 0
  
  for i in range(0, 32):
    for j in range(0, 32):
      extend_image[i+P][j+P][0] = img[i][j][0];
      extend_image[i+P][j+P][1] = img[i][j][1];
      extend_image[i+P][j+P][2] = img[i][j][2];


  #print the image
  #for j in range(0, 34):
  #    for k in range(0, 34):
  #        print extend_image[j][k][0],
  #            
  #    print "\n"

  for i in range(0,output_size):
    for j in range(0,output_size):
      if i == 0 and j == 0:
        conv_data = extend_image[i:F+i,j:F+j,:].reshape(F*F*3,1)
        #print extend_image[i:F+i,j:F+j].reshape(F*F*3,1)
      else:
        conv_data = np.hstack((conv_data, extend_image[i:F+i,j:F+j,:].reshape(F*F*3,1)))


  return conv_data
 
def pooling(pooling_data, num_images, img_size):
  row = pooling_data.shape[0]  # 12
  col = pooling_data.shape[1]/(num_images*4) # 225
  pooling_batch = np.empty((row, col))
  pooling_index = np.empty((row, col))

  #print '# of images:', num_images
  #print 'row:', row
  #print 'col:', col

  img_2d_size = img_size*img_size
  half_img_size = img_size/2
  #print img_size

  for k in range(0, num_images):
    pooling_result = np.empty((row, col))
    tmp_pooling_index = np.empty((row, col), dtype=int)

    #print k, 'th image'
    for i in range (0,row):
      offset = 0
      for j in range(0, col):

        if j%half_img_size == 0 and j != 0:
          offset += img_size*2

        top_index    = k*img_2d_size+offset+(j%half_img_size)*2
        bottom_index = k*img_2d_size+offset+img_size+(j%half_img_size)*2

        top_part = pooling_data[i][top_index:top_index+2]
        bottom_part = pooling_data[i][bottom_index:bottom_index+2]
      
        #print 'top:', top_part
        #print top_index
        #print 'bottom:', bottom_part
        #print bottom_index
        pooling_result[i][j] = max(top_part+bottom_part)
        max_index = np.argmax(top_part+bottom_part)

        if max_index < 2:
          tmp_pooling_index[i][j] = np.argmax(pooling_data[i][top_index:top_index+2])
        else:
          tmp_pooling_index[i][j] = np.argmax(pooling_data[i][bottom_index:bottom_index+2])
        #pooling_result[i][j] = max(pooling_data[i][k*900j:4*(j+1)])
        #tmp_pooling_index[i][j] = np.argmax(pooling_data[k][i][j:4*(j+1)]) 


    if k == 0:
      pooling_batch = pooling_result
      pooling_index = tmp_pooling_index
    else:
      pooling_batch = np.vstack((pooling_batch,pooling_result))
      pooling_index = np.vstack((pooling_index,tmp_pooling_index))


  pooling_batch = pooling_batch.reshape(num_images, row*col)
  pooling_index = pooling_index.reshape(num_images, row*col)
  #pooling_batch = pooling_batch.reshape(pooling_data.shape[0], row, col/4)

  return pooling_batch.T, pooling_index.T

def batchRead(input_data, input_label):
  batch_idx = np.random.randint(0,len(input_data),mini_batch)

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
  
  #print 'before', img_batch.shape
  img_batch = img_batch.reshape(mini_batch,32,32,3)
  #print 'after', img_batch.shape

  return img_batch, label_batch


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
      label_batch = np.hstack((label_batch, input_label[i]))
  
  #print 'before', img_batch.shape
  img_batch = img_batch.reshape(len(input_data),32,32,3)
  #print 'after', img_batch.shape

  return img_batch, label_batch




#def batchIm2Col(input_data, W, F, S, P):
def batchIm2Col(input_data):
  ReLU_size = (W-F+2*P)/S + 1 #30
  col_batch = np.empty([F*F*3, ReLU_size*ReLU_size])
  for i in range(0, input_data.shape[0]):
    #Convolutional layer
    col_img = im2col(input_data[i])
    #col_img = im2col(input_data[i], W, F, S, P)
                                           #    32  |    32  |  32   

    if i == 0:
      col_batch = col_img
    else:
      col_batch = np.hstack((col_batch, col_img))

  #col_batch = col_batch.reshape(F*F*3,ReLU_size*ReLU_size*mini_batch)
  #print 'ReLU size:', ReLU_size
  #print 'col_batch:', col_batch.shape

  return col_batch


def ReLULayer(conv_data, W, b):
  ReLU_data = np.empty([W.shape[0], conv_data.shape[1]])

  for i in range(0, conv_data.shape[0]):
    conv_img = conv_data[i]

    if i == 0:
      ReLU_data = np.maximum(0, np.dot(W,conv_img) + b)
    else:
      ReLU_data = np.hstack((ReLU_data, np.maximum(0, np.dot(W,conv_img) + b)))

  ReLU_data = ReLU_data.reshape(conv_data.shape[0], W.shape[0], conv_data.shape[2])

  return ReLU_data



def updateGradientReLU(dReLU, dpool, pool_index):
  row = dReLU.shape[0]  # 12
  col = dReLU.shape[1]  # 225000

  for i in range (0,row):
    for j in range (0,col/4):
      dReLU[i][pool_index[i][j]] = dpool[i][j]




  return dReLU

#def backpg2ReLU(dReLU, ReLU):
#  for i in range(0, dReLU.shape[0]):
#    dReLU[i][ReLU[i] <= 0] = 0
#
#  return dReLU
  

def updateWeights(col_images, dReLU):
  #print dReLU.shape[1]
  #print col_images.shape[1]
  row = dReLU.shape[1]
  col = col_images.shape[1] 

  dW = np.zeros((row, col))
  #dW = np.zeros((12,75))
  for i in range(0, col_images.shape[0]):
    dW += np.dot(dReLU[i], col_images[i].T)

  return dW


def updateBias(dReLU):

  db = np.zeros((1, dReLU.shape[1]))
  #dW = np.zeros((12,75))
  for i in range(0, dReLU.shape[0]):
    #print dReLU[i].shape
    #print np.sum(dReLU[i].T, axis=0, keepdims=True).shape
    db += np.sum(dReLU[i].T, axis=0, keepdims=True)

  return db


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
  mini_batch = 128
  num_filters = 16
  K = 10 # number of classes
  W = 32 # the size of image W X W
  F = 5  # the size of filter F X F
  S = 1  # stride
  P = 2  # padding
  ReLU_1_size = (W-F+2*P)/S + 1

  reg = 1e-3 # regularization strength
  step_size = 5e-5


  # initialize parameters randomly
  W1 = 0.01*np.random.randn(num_filters,F*F*3)*(np.sqrt(2./(F*F*3)))
  b1 = np.zeros((num_filters,1))

  W2 = 0.01*np.random.randn(K,num_filters*ReLU_1_size*ReLU_1_size/4)*(np.sqrt(2./(num_filters*ReLU_1_size*ReLU_1_size/4)))
  b2 = np.zeros((K,1))

  for itr in xrange(10):
    X, y = batchRead(tr_data10, tr_labels10)
    #X, y = batchRead(tr_data10, tr_labels10, W, F, S, P)

    #Convolutional layer
    col_images = batchIm2Col(X)
    #col_images = batchIm2Col(X, W, F, S, P)
    #print col_images.shape


    #ReLU layer
    #ReLU_1 = ReLULayer(col_images, W1, b1)
    ReLU_1 = np.maximum(0, np.dot(W1,col_images) + b1)
    #print 'ReLU layer size: ',  ReLU_1.shape

    #Pooling layer
    pool_1, pool_1_index = pooling(ReLU_1, mini_batch, ReLU_1_size)
    #print 'Pooling layer size: ',  pool_1.shape
    #print 'Pooling index size: ',  pool_1_index.shape
    #print 'Pooling index: ',  pool_1_index.shape

    #print W2.shape
    #print pool_1
    scores = np.dot(W2, pool_1) + b2
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs.T[range(mini_batch),y])
    data_loss = np.sum(corect_logprobs)/mini_batch
    reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss

    predicted_class = np.argmax(scores, axis=0)
 
    if itr % 5 == 0:
      print "iteration %d: loss %f training accuracy: %.2f" % (itr, loss, np.mean(predicted_class == y))

    # compute the gradient on scores
    dscores = probs.T #[mini-batch X K]
    dscores[range(mini_batch),y] -= 1
    dscores /= mini_batch

    # backpropate the gradient to the parameters (W,b)
    # first backprop into parameters W2 and b2
    dW2 = np.dot(dscores.T, pool_1.T)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into pooling layer
    dpool_1 = np.dot(dscores, W2)  # (mini-batch, K) (K,2700)
    dReLU_1 = np.zeros(ReLU_1.shape)


    #print 'dReLU_1 size:', dReLU_1.shape
    #print 'dpool_1 size:', dpool_1.shape
    #print 'pool_1_index size:', pool_1_index.shape
    dReLU_1 = updateGradientReLU(dReLU_1, #(12,22500)
                                 dpool_1.reshape(mini_batch*ReLU_1_size*ReLU_1_size/4, num_filters).T, #(12,56250)
                                 pool_1_index.reshape(mini_batch*ReLU_1_size*ReLU_1_size/4, num_filters).T)
    #print dReLU_1
    #print 'dReLU size: ',  dReLU_1.shape

    # backprop the ReLU non-linearity
    #dReLU_1 = backpg2ReLU(dReLU_1, ReLU_1)
    dReLU_1[ReLU_1 <= 0] = 0
    #print dReLU_1

    #dW1 = updateWeights(col_images, dReLU_1)
    #db1 = updateBias(dReLU_1)
    dW1 = np.dot(dReLU_1, col_images.T) 
    db1 = np.sum(dReLU_1.T, axis=0, keepdims=True)
    #print dW1


    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg * W1

    param_scale_1 = np.linalg.norm(W1.ravel())
    param_scale_2 = np.linalg.norm(W2.ravel())

    #print 'param_scale_1:', param_scale_1
    #print 'param_scale_2:', param_scale_2

    update_1 = step_size * dW1
    update_2 = step_size * dW2


    update_scale_1 = np.linalg.norm(update_1.ravel())
    #print  update_2
    update_scale_2 = np.linalg.norm(update_2.ravel())
    #print 'update_scale_2:', update_scale_2
    #print 'update_scale_1:', update_scale_1

    #print 'ratio w1:', update_scale_1/ param_scale_1
    #print 'ratio w2:', update_scale_2/ param_scale_2
    # perform a parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1.T
    W2 += -step_size * dW2
    b2 += -step_size * db2.T



  ###########################################
  #            Evaluation                   #
  ###########################################
  for itr in xrange(5):
    # evaluate test set accuracy
    X, y_test = batchRead(te_data10, te_labels10)

    #Convolutional layer
    col_images = batchIm2Col(X)
    ReLU_1 = np.maximum(0, np.dot(W1,col_images) + b1)
    pool_1, pool_1_index = pooling(ReLU_1, mini_batch, ReLU_1_size)
    scores = np.dot(W2, pool_1) + b2
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    predicted_class = np.argmax(scores, axis=0)
    print 'test accuracy: %.2f' % (np.mean(predicted_class == y_test))
 
