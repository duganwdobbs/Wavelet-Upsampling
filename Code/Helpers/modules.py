# A basic Module Functional Library

import tensorflow as tf
from Helpers import ops,util

def Dense_Block(net,level,features = 12, kernel = 3, kmap = 5):
  with tf.variable_scope("Dense_Block_%d"%level) as scope:
    # If net is multiple tensors, concat them.
    net = ops.delist(net)
    # Setup a list to store map outputs.
    outs = []

    # Dummy variable for training and trainable atm
    training = True
    trainable = True

    for n in range(kmap):
      out  = ops.delist([net,outs]) if n > 0 else net
      out  = ops.conv2d(out,filters=features,kernel=kernel,stride=1,activation=None,padding='SAME',name = '_map_%d'%n)
      out  = ops.batch_norm(out,training,trainable)
      out  = tf.nn.leaky_relu(out)
      outs = tf.concat([outs,out],-1,name='%d_cat'%n) if n > 0 else out

    return outs

def Transition_Down(net,stride,level):
  with tf.variable_scope("Transition_Down_%d"%level) as scope:
    net  = ops.delist(net)
    # Differentiation from 100 Layers: No 1x1 convolution here.
    net  = ops.avg_pool(net,stride,stride)
    return net

def Transition_Up(net,stride,level):
  with tf.variable_scope("Transition_Up_%d"%level) as scope:
    net = ops.deconvxy(net,stride)
    return net

def Encoder(net,kmap,feature,stride,level):
  '''
  This is our encoder function helper. It receives a good number of variables,
  then constructs one level of our encoding pathway. This involves a dense
  block, then transition down.
  Parameters:
    net           : This is the hidden state of the network to upsample
    kmap          : This is the number of kmaps in the dense block
    feature       : This is the feature growth for the dense block
    stride        : This is the striding amount to pool
    level         : This is an int referring to the level number

  Returns:
    net           : The downsampled hidden network state
    skip          : The skip connection output from pre-pooling
  '''
  with tf.variable_scope("Encoder_%d"%level) as scope:
    skip  = Dense_Block(net,level,features = feature,kmap = kmap)
    down  = Transition_Down([net,skip],stride,level)

  return skip,down

def Decoder(net,skip,kmap,feature,stride,level,residual_conn = True):
  '''
  This is our decoder function helper. It receives a good number of variables,
  then constructs one level of our decoding pathway. This involves a upsample,
  dense block, then residual connection.
  Parameters:
    net           : This is the hidden state of the network to upsample
    skip          : This is the skip connection from the downsample path
    kmap          : This is the number of kmaps in the dense block
    feature       : This is the feature growth for the dense block
    stride        : This is the striding amount to upsample
    level         : This is an int referring to the level number
    residual_conn : This is if we use a residual connection

  Returns:
    net           : The output hidden network state
  '''
  with tf.variable_scope("Decoder_%d"%level) as scope:
    net   = Transition_Up(net,stride,level)
    resid = Dense_Block([net,skip],level,features = feature,kmap = kmap)
    net   = ops.delist([net,resid])

  return net

def Encoder_Decoder(net,out_features = None,name="Encoder_Decoder"):
  if out_features is None:
    b,h,w,c = net.get_shape().as_list()
    out_features = c

  with tf.variable_scope(name) as scope:
    trainable   = True
    kmaps       = [ 3, 5]
    features    = [ 4, 6]
    strides     = [ 2, 3]
    skips       = []

    for x in range(len(strides)):
      level = x+1
      stride = strides[x]
      kmap   = kmaps[x]
      feature= features[x]

      skip,net = Encoder(net,kmap,feature,stride,x+1)

      skips.append(skip)

    for x in range(1,len(strides)+1):
      skip   = skips[-x]
      stride = strides[-x]
      kmap   = kmaps[-x]
      feature= features[-x]

      net = Decoder(net,skip,kmap,feature,stride,len(strides)+1-x)

    net = ops.conv2d(net,out_features,3,name='Decomp_Formatter',activation = None)
    return net
