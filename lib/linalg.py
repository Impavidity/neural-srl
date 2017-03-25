#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#***************************************************************
sig_const = np.arctanh(1/3)
tanh_const = np.arctanh(np.sqrt(1/3))

def tanh(x):
  return tf.tanh(x)
def sigmoid(x):
  return (tf.tanh(x)+1)/2

#===============================================================
def orthonormal_initializer(input_size, output_size):
  """"""
  
  print(tf.get_variable_scope().name)
  I = np.eye(output_size)
  lr = .1
  eps = .05/(output_size + input_size)
  success = False
  while not success:
    Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    for i in xrange(100):
      QTQmI = Q.T.dot(Q) - I
      loss = np.sum(QTQmI**2 / 2)
      Q2 = Q**2
      Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
      if np.isnan(Q[0,0]):
        lr /= 2
        break
    if np.isfinite(loss) and not Q[0,0] > 1e6:
      success = True
  print('Orthogonal pretrainer loss: %.2e' % loss)
  return Q.astype(np.float32)

#===============================================================
def linear(inputs, output_size, add_bias=True, n_splits=1, initializer=None, scope=None, moving_params=None):
  """"""
  
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  output_size *= n_splits
  
  with tf.variable_scope(scope or 'Linear'):
    # Reformat the input
    total_input_size = 0
    shapes = [a.get_shape().as_list() for a in inputs]
    for shape in shapes:
      total_input_size += shape[-1]
    input_shape = tf.shape(inputs[0])
    output_shape = []
    for i in xrange(len(shapes[0])):
      output_shape.append(input_shape[i])
    output_shape[-1] = output_size
    #print(output_shape)
    output_shape = tf.pack(output_shape)
    for i, (input_, shape) in enumerate(zip(inputs, shapes)):
      inputs[i] = tf.reshape(input_, [-1, shape[-1]])
    concatenation = tf.concat(1, inputs)
    
    # Get the matrix
    if initializer is None and moving_params is None:
      mat = orthonormal_initializer(total_input_size, output_size//n_splits)
      mat = np.concatenate([mat]*n_splits, axis=1)
      initializer = tf.constant_initializer(mat)
    matrix = tf.get_variable('Weights', [total_input_size, output_size], initializer=initializer)
    if moving_params is not None:
      matrix = moving_params.average(matrix)
    else:
      tf.add_to_collection('Weights', matrix)
    
    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer)
      if moving_params is not None:
        bias = moving_params.average(bias)
    else:
      bias = 0
    
    # Do the multiplication
    new = tf.matmul(concatenation, matrix) + bias
    new = tf.reshape(new, output_shape)
    if n_splits > 1:
      return tf.split(len(new.get_shape().as_list())-1, n_splits, new)
    else:
      return new

#============================================
def my_linear(inputs, output_size, add_bias=True, n_splits=1, initializer=None, scope=None, moving_params=None):
  """"""

  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  output_size *= n_splits

  with tf.variable_scope(scope or 'Linear'):
    # Reformat the input
    total_input_size = 0
    shapes = [a.get_shape().as_list() for a in inputs]
    for shape in shapes:
      total_input_size += shape[-1]
    input_shape = tf.shape(inputs[0])
    output_shape = []
    for i in xrange(len(shapes[0])):
      output_shape.append(input_shape[i])
    output_shape[-1] = output_size
    # print(output_shape)
    output_shape = tf.pack(output_shape)
    for i, (input_, shape) in enumerate(zip(inputs, shapes)):
      inputs[i] = tf.reshape(input_, [-1, shape[-1]])
    concatenation = tf.concat(1, inputs)

    # Get the matrix
    '''
    if initializer is None and moving_params is None:
      mat = orthonormal_initializer(total_input_size, output_size // n_splits)
      mat = np.concatenate([mat] * n_splits, axis=1)
      initializer = tf.constant_initializer(mat)
    '''
    matrix = tf.get_variable('Weights', [total_input_size, output_size])
    if moving_params is not None:
      matrix = moving_params.average(matrix)
    else:
      tf.add_to_collection('Weights', matrix)

    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer)
      if moving_params is not None:
        bias = moving_params.average(bias)
    else:
      bias = 0

    # Do the multiplication
    new = tf.matmul(concatenation, matrix) + bias
    new = tf.reshape(new, output_shape)
    if n_splits > 1:
      return tf.split(len(new.get_shape().as_list()) - 1, n_splits, new)
    else:
      return new


#===============================================================
def bilinear(inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False, initializer=None, scope=None, moving_params=None):
  """"""
  
  with tf.variable_scope(scope or 'Bilinear'):
    # Reformat the inputs
    ndims = len(inputs1.get_shape().as_list())
    inputs1_shape = tf.shape(inputs1)
    inputs1_bucket_size = inputs1_shape[ndims-2]
    inputs1_size = inputs1.get_shape().as_list()[-1]
    
    inputs2_shape = tf.shape(inputs2)
    inputs2_bucket_size = inputs2_shape[ndims-2]
    inputs2_size = inputs2.get_shape().as_list()[-1]
    output_shape = []
    batch_size = 1
    for i in xrange(ndims-2):
      batch_size *= inputs1_shape[i]
      output_shape.append(inputs1_shape[i])
    output_shape.append(inputs1_bucket_size)
    output_shape.append(output_size)
    output_shape.append(inputs2_bucket_size)
    output_shape = tf.pack(output_shape)
    inputs1 = tf.reshape(inputs1, tf.pack([batch_size, inputs1_bucket_size, inputs1_size]))
    inputs2 = tf.reshape(inputs2, tf.pack([batch_size, inputs2_bucket_size, inputs2_size]))
    if add_bias1:
      inputs1 = tf.concat(2, [inputs1, tf.ones(tf.pack([batch_size, inputs1_bucket_size, 1]))])
    if add_bias2:
      inputs2 = tf.concat(2, [inputs2, tf.ones(tf.pack([batch_size, inputs2_bucket_size, 1]))])
    
    # Get the matrix
    if initializer is None and moving_params is None:
      mat = orthonormal_initializer(inputs1_size+add_bias1, inputs2_size+add_bias2)[:,None,:]
      mat = np.concatenate([mat]*output_size, axis=1)
      initializer = tf.constant_initializer(mat)
    weights = tf.get_variable('Weights', [inputs1_size+add_bias1, output_size, inputs2_size+add_bias2], initializer=initializer)
    if moving_params is not None:
      weights = moving_params.average(weights)
    else:
      tf.add_to_collection('Weights', weights)
    
    # Do the multiplications
    # (bn x d) (d x rd) -> (bn x rd)
    lin = tf.matmul(tf.reshape(inputs1, [-1, inputs1_size+add_bias1]),
                        tf.reshape(weights, [inputs1_size+add_bias1, -1]))
    # (b x nr x d) (b x n x d)T -> (b x nr x n)
    bilin = tf.batch_matmul(tf.reshape(lin, tf.pack([batch_size, inputs1_bucket_size*output_size, inputs2_size+add_bias2])),
                                   inputs2, adj_y=True)
    # (bn x r x n)
    bilin = tf.reshape(bilin, tf.pack([-1, output_size, inputs2_bucket_size]))
    # (b x n x r x n)
    bilin = tf.reshape(bilin, output_shape)
    
    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer)
      if moving_params is not None:
        bias = moving_params.average(bias)
      bilin += tf.expand_dims(bias, 1)
    
    return bilin

#===============================================================
def diagonal_bilinear(inputs1, inputs2, output_size, add_bias1=True, add_bias2=True, initializer=None, scope=None, moving_params=None):
  """"""
  
  with tf.variable_scope(scope or 'DiagonalBilinear'):
    # Reformat the inputs
    ndims = len(inputs1.get_shape().as_list())
    inputs1_shape = tf.shape(inputs1)
    inputs1_bucket_size = inputs1_shape[ndims-2]
    inputs1_size = inputs1.get_shape().as_list()[-1]
    
    inputs2_shape = tf.shape(inputs2)
    inputs2_bucket_size = inputs2_shape[ndims-2]
    inputs2_size = inputs2.get_shape().as_list()[-1]
    output_shape = []
    batch_size = 1
    for i in xrange(ndims-2):
      batch_size *= inputs1_shape[i]
      output_shape.append(inputs1_shape[i])
    output_shape.append(inputs1_bucket_size)
    output_shape.append(output_size)
    output_shape.append(inputs2_bucket_size)
    output_shape = tf.pack(output_shape)
    inputs1 = tf.reshape(inputs1, tf.pack([batch_size, inputs1_bucket_size, inputs1_size]))
    inputs2 = tf.reshape(inputs2, tf.pack([batch_size, inputs2_bucket_size, inputs2_size]))
    
    # Get the matrix
    if initializer is None and moving_params is None:
      initializer = tf.ones_initializer
    weights = tf.get_variable('Weights', [output_size, inputs1_size], initializer=initializer)
    if moving_params is not None:
      weights = moving_params.average(weights)
    else:
      tf.add_to_collection('Weights', weights)
    
    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer)
      if moving_params is not None:
        bias = moving_params.average(bias)
      bias = tf.reshape(bias, [-1,1])
    else:
      bias = 0
    
    # Do the multiplications
    # (bn x 1 x d) (r x d) -> (bn x r x d)
    lin = tf.reshape(inputs1, [-1, 1, inputs1_size]) * weights
    # (b x nr x d) (b x n x d)T -> (b x nr x n)
    bilin = tf.batch_matmul(tf.reshape(lin, tf.pack([batch_size, inputs1_bucket_size*output_size, inputs2_size])),
                                   inputs2, adj_y=True)
    # (bn x r x n)
    bilin = tf.reshape(bilin, tf.pack([-1, output_size, inputs2_bucket_size])) + bias
    # (b x n x r x n)
    bilin = tf.reshape(bilin, output_shape)
    
    if add_bias1:
      with tf.variable_scope('Input1_Biases'):
        inputs1.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(inputs1_size)])
        bilin += tf.expand_dims(linear(inputs1, output_size, add_bias=False, moving_params=moving_params), 3)
    if add_bias2:
      with tf.variable_scope('Input2_Biases'):
        inputs2.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(inputs2_size)])
        bilin += tf.expand_dims(tf.transpose(linear(inputs2, output_size, add_bias=False, moving_params=moving_params), [0, 2, 1]), 1)
    
    return bilin
  
#===============================================================
def layer_norm(inputs, beta_start=0, gamma_start=1, scope=None, moving_params=None):
  """"""
  
  with tf.variable_scope(scope or "Layer_norm"):
    gamma = tf.get_variable('Gamma', shape=[],
                            initializer=tf.constant_initializer(gamma_start))
    beta = tf.get_variable('Beta', shape=[],
                            initializer=tf.constant_initializer(beta_start))
    if moving_params is not None:
      gamma = moving_params.average(gamma)
      beta = moving_params.average(beta)
    mean, var = tf.nn.moments(inputs, 1, keep_dims=True)
    inputs = gamma * (inputs-mean) / tf.sqrt(var+self.epsilon) + beta
    return inputs
  
#***************************************************************
if __name__ == '__main__':
  """"""
  
  x1 = tf.Variable(np.random.randn(5,5).astype(np.float32))
  x2 = tf.Variable(np.random.randn(5,2).astype(np.float32))
  z = linear([x1, x2], 10)
  zz = bilinear(x1, x2, 10)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(z)
    sess.run(zz)
