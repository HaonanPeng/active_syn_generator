import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *


image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter
seed = 23

def batchnorm(input_,is_train=False,name="batchnorm"):
    with tf.variable_scope(name):
        normalized = tf.layers.batch_normalization(input_, training=is_train)
        return normalized

def conv2d(input_, output_dim, ksize=3, stride=2, stddev=0.02,name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [ksize, ksize, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev, seed=seed))
              
    conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
#    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    conv = tf.nn.bias_add(conv, biases)

    return conv

def conv2d_dilated(input_, output_dim, ksize=3, rate=2, stddev=0.02,name="conv2d_dilated"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [ksize, ksize, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev, seed=seed))
              
    conv = tf.nn.atrous_conv2d(input_,w,rate=rate,padding="SAME")


    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
#    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    conv = tf.nn.bias_add(conv, biases)

    return conv

def deconv2d(input_, output_shape,
       ksize=5, stride=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [ksize, ksize, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.truncated_normal_initializer(stddev=stddev, seed=seed))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, stride, stride, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, stride, stride, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)
  
def prelu(x, name="prelu"):
  with tf.variable_scope(name):
    alpha = tf.get_variable("prelu", shape=x.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)
    
    
def relu(x, name="relu"):
  return tf.maximum(x, 0)
  
def separable_conv2d(input_, output_dim, ksize=3, stride=1,rate=1, stddev=0.02,name=''):
    with tf.variable_scope(name+"_separable_conv2d"):
        in_chns = input_.get_shape()[3].value
        w_depth = tf.get_variable('w_depth', [ksize,ksize,in_chns,1],initializer=tf.truncated_normal_initializer(stddev=stddev, seed=seed))
        w_point = tf.get_variable('w_point', [1,1,in_chns,output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev, seed=seed))
        conv = tf.nn.separable_conv2d( input_,
                        depthwise_filter = w_depth,
                        pointwise_filter = w_point,
                        strides = [1,stride,stride,1],
                        padding="SAME",
                        rate=[rate,rate],
                        name="sep_conv")
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv, biases)

    return output


def atrous_spatial_pyramid_pooling(input_, output_stride=16, depth=256,is_train=False,dropout=False,keep_prob=1.0):
  """Atrous Spatial Pyramid Pooling.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.

    depth: The depth of the ResNet unit output.
  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):

    atrous_rates = [2,4]#[6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
    # the rates are doubled when output stride = 8.
    h1 = conv2d(input_, depth, ksize=1, stride=1, name="conv1")
    h1 = tf.nn.relu(batchnorm(h1,is_train,'bn1'))
    
    h2 = conv2d_dilated(input_, depth, ksize=3,rate=atrous_rates[0], name="conv3_1")
    h2 = tf.nn.relu(batchnorm(h2,is_train,'bn2'))
    
    h3 = conv2d_dilated(input_, depth, ksize=3,rate=atrous_rates[1], name="conv3_2")
    h3 = tf.nn.relu(batchnorm(h3,is_train,'bn3'))
    
    
    # (b) the image-level features
    input_size = tf.shape(input_)[1:3]
    h0 = tf.reduce_mean(input_, [1, 2], name='global_average_pooling', keepdims=True)
    h0 = conv2d(h0, depth, ksize=1, stride=1, name="conv1_pool")
    h0 = tf.nn.relu(batchnorm(h0,is_train,'bn_gap'))
    h0 = tf.image.resize_bilinear(h0, input_size, name='upsample')
    
    
    h = tf.concat([h0,h1,h2,h3],axis=3)

    h = conv2d(h, depth, ksize=1, stride=1, name="conv1_out")
    h = tf.nn.relu(batchnorm(h,is_train,'bn_out'))
    
    if dropout:
        h = tf.nn.dropout(h,keep_prob,seed=seed) 
    
    return h
    

