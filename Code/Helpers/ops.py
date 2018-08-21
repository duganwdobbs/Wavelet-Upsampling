import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('conv_scope',0,'Incrementer for convolutional scopes')
flags.DEFINE_integer('bn_scope'  ,0,'Incrementer for batch norm scopes')

bn_scope = 0

def init_scope_vars():
  FLAGS.conv_scope = 0
  FLAGS.bn_scope = 0

def delist(net):
  if type(net) is list:
    net = tf.concat(net,-1,name = 'cat')
  return net

def im_norm(net):
  with tf.variable_scope('Image_Normalization') as scope:
    net = tf.cast(net,tf.float32)
    net = net / 255 - 1
    return net

def lrelu(x):
  with tf.variable_scope('lrelu') as scope:
    if x.dtype is not tf.complex64:
      return tf.nn.leaky_relu(x)
    else:
      return x

def relu(x):
  with tf.variable_scope('relu') as scope:
    if x.dtype is not tf.complex64:
      return tf.nn.relu(x)
    else:
      return x

def linear(x):
  return x

def conv2d(net, filters, kernel = 3, stride = 1, dilation_rate = 1, activation = relu, padding = 'SAME', trainable = True, name = None, reuse = None):
  net = tf.layers.conv2d(delist(net),filters,kernel,stride,padding,dilation_rate = dilation_rate, activation = activation,trainable = trainable, name = name, reuse = reuse)
  return net

def bn_conv2d(net, training, filters, kernel = 3, stride = 1, dilation_rate = 1, activation = lrelu, use_bias = False, padding = 'SAME', trainable = True, name = None, reuse = None):
  with tf.variable_scope('BN_Conv_%d'%(FLAGS.conv_scope)) as scope:
    FLAGS.conv_scope+=1
    net = conv2d(delist(net), filters, kernel, stride, dilation_rate, activation, padding, trainable, name, reuse)
    net = batch_norm(net,training,trainable,activation)
    return net

def batch_norm(net,training,trainable,activation = None):
  with tf.variable_scope('Batch_Norm_%d'%(FLAGS.bn_scope)):
    FLAGS.bn_scope = FLAGS.bn_scope + 1
    net = tf.layers.batch_normalization(delist(net),training = training, trainable = trainable)

    if activation is not None:
      net = activation(net)

    return net

def avg_pool(net, kernel = 3, stride = 1, padding = 'SAME', name = None):
  return tf.layers.average_pooling2d(net,kernel,stride,padding=padding,name=name)

def max_pool(net, kernel = 3, stride = 3, padding = 'SAME', name = None):
  return tf.layers.max_pooling2d(net,kernel,stride,padding=padding,name=name)

def conv2d_trans(net, filters, kernel, stride, activation = relu,padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d_transpose(net,filters,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)

def deconv(net, filters = 3, kernel = 3, stride = 2, activation = lrelu,padding = 'SAME', trainable = True, name = None):
  net = tf.layers.conv2d_transpose(net,filters,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)
  return net

def deconvxy(net, stride = 2,filters = None, activation = None,padding = 'SAME', trainable = True, name = 'Deconv_xy'):
  with tf.variable_scope(name) as scope:

    net = delist(net)

    kernel = stride+1

    if filters is None:
      filters = int(net.shape[-1].value / stride)

    netx = deconv(net, filters  , kernel = kernel, stride = (stride,1), activation = activation, name = "x",  trainable = trainable)
    nety = deconv(net, filters  , kernel = kernel, stride = (1,stride), activation = activation, name = "y",  trainable = trainable)

    # filters = int(filters / stride)

    netx = deconv(netx, filters  , kernel = kernel, stride = (1,stride), activation = activation, name = "xy"  , trainable = trainable)
    nety = deconv(nety, filters  , kernel = kernel, stride = (stride,1), activation = activation, name = "yx"  , trainable = trainable)
    netz = deconv(net , filters  , kernel = kernel, stride = stride    , activation = activation, name = 'xyz' , trainable = trainable)

    net  = tf.concat((netx,nety),-1)

    return net

def dense_block(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
                activation = relu, padding = 'SAME', trainable = True,
                name = 'Dense_Block', prestride_return = True,use_max_pool = True,
                in_return = True):
  with tf.variable_scope(name) as scope:

    net = delist(net)
    outs = []

    for n in range(kmap):
      out = batch_norm(delist([net,outs]),training,trainable) if n > 0 else batch_norm(net,training,trainable)
      out = relu(out)
      out = conv2d(out,filters=filters,kernel=kernel,stride=1,activation=None,padding=padding,trainable=trainable,name = '_map_%d'%n)
      out = tf.nn.dropout(out,FLAGS.keep_prob)
      outs = delist([outs,out]) if n > 0 else out

    if in_return:
      net = delist([net,outs])

    if stride > 1:
      prestride = net
      if use_max_pool:
        net = max_pool(net,stride,stride)
      else:
        net = avg_pool(net,stride,stride)
      if prestride_return:
        return prestride, net
      else:
        return net

    else:
      return net

def dense_block_out(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
                  activation = relu, padding = 'SAME', trainable = True,
                  name = 'Dense_Block', prestride_return = False,use_max_pool = True):
    with tf.variable_scope(name) as scope:

      net = delist(net)
      outs = []

      for n in range(kmap):
        out = batch_norm(net,training,trainable)
        out = activation(out)
        out = conv2d(out,filters=filters,kernel=kernel,stride=1,activation=None,padding=padding,trainable=trainable,name = '_map_%d'%n)
        out = tf.nn.dropout(out,FLAGS.keep_prob)
        outs.append(out)
        net = tf.concat([net,out],-1,name = '%d_concat'%n)

      if stride > 1:
        prestride = net
        if use_max_pool:
          net = max_pool(net,stride,stride)
        else:
          net = avg_pool(net,stride,stride)
        if prestride_return:
          return prestride, net
        else:
          return net

      else:
        return delist(outs)

def atrous_block(net,training,filters = 8,kernel = 3,dilation = 1,kmap = 2,stride = 1,activation = relu,trainable = True,name = 'Atrous_Block'):
  newnet = []
  with tf.variable_scope(name) as scope:
    for x in range(dilation,kmap * dilation,dilation):
      # Reuse and not trainable if beyond the first layer.
      re = True  if x > dilation else None
      tr = False if x > dilation else trainable

      with tf.variable_scope("ATROUS",reuse = tf.AUTO_REUSE) as scope:
        # Total Kernel visual size: Kernel + ((Kernel - 1) * (Dilation - 1))
        # At kernel = 9 with dilation = 2; 9 + 8 * 1, 17 px
        layer = conv2d(net,filters = filters, kernel = kernel, dilation_rate = x, activation = None,reuse = re,trainable = tr,padding='SAME')
        newnet.append(layer)

    net = delist(newnet)
    net = bn_conv2d(net,training,filters = filters,kernel = stride,stride = stride,trainable = trainable,name = 'GradientDisrupt',activation = activation)
    return net


# Defines a function to output the histogram of trainable variables into TensorBoard
def hist_summ():
  for var in tf.trainable_variables():
    tf.summary.histogram(var.name,var)

def cmat(labels_flat,logits_flat):
  with tf.variable_scope("Confusion_Matrix") as scope:
    label_1d  = tf.reshape(labels_flat, (FLAGS.batch_size, FLAGS.imgW * FLAGS.imgH))
    logit_1d = tf.reshape(logits_flat, (FLAGS.batch_size, FLAGS.imgW * FLAGS.imgH))
    cmat_sum = tf.zeros((FLAGS.num_classes,FLAGS.num_classes),tf.int32)
    for i in range(FLAGS.batch_size):
      cmat = tf.confusion_matrix(labels = label_1d[i], predictions = logit_1d[i], num_classes = FLAGS.num_classes)
      cmat_sum = tf.add(cmat,cmat_sum)
    return cmat_sum

def l2loss(loss = None,loss_vars = None,l2 = None):
  if l2 is None:
    l2 = FLAGS.l2_loss
  if l2:
    with tf.variable_scope("L2_Loss") as scope:
      loss_vars = tf.trainable_variables() if loss_vars is None else loss_vars
      l2 = tf.add_n([tf.nn.l2_loss(var) for var in loss_vars if 'bias' not in var.name])
      l2 = tf.scalar_mul(.0002,l2)
      tf.summary.scalar('L2_Loss',l2)
      if loss is not None:
        loss = tf.add(loss,l2)
        tf.summary.scalar('Total_Loss',loss)
      else:
        loss = l2
  return loss

# A log loss for using single class heat map
def log_loss(labels,logits):
  with tf.variable_scope('Log_Loss') as scope:
    loss = tf.losses.log_loss(labels,logits)
  tf.summary.scalar('Log_Loss',loss)
  loss = l2loss(loss)
  return loss

def xentropy_loss(labels,logits,loss_vars = None,l2 = None,class_weights = None,name = 'Xent_Loss'):
  with tf.variable_scope(name) as scope:
    labels = tf.cast(labels,tf.int32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,logits = logits)
    tf.summary.scalar('XEnt_Loss',tf.reduce_mean(loss))

    if class_weights is not None:
      # [tf.summary.scalar("Weight_%d"%x,class_weights[x]) for x in range(FLAGS.num_classes)]
      weights = tf.reduce_sum(class_weights * tf.one_hot(labels,FLAGS.num_classes), axis=-1)
      loss = loss * weights
      tf.summary.scalar('Weighted_Loss',tf.reduce_mean(loss))

    loss = tf.reduce_mean(loss)

    loss = l2loss(loss,loss_vars,l2)
    return loss

def focal_loss(labels,logits,tuning = 2,alpha = None):
  with tf.variable_scope("AlphaFocalLoss") as scope:
    logits  = tf.nn.softmax(logits,-1)
    oh_lab  = tf.one_hot(labels,FLAGS.num_classes)
    eps     = 1e-10
    with tf.variable_scope("AlphaGeneration") as scope:
      if alpha is None:
        alpha   = [1 / (tf.reduce_mean(oh_lab[:,:,:,x]) * FLAGS.num_classes + eps) for x in range(FLAGS.num_classes)]
        alpha   = tf.stack(alpha)
      step    = tf.train.get_global_step()
      decay   = tf.train.exponential_decay(.90,step,100,.99,staircase=True)
      tf.summary.scalar("AlphaDecay",decay)
      alpha   = alpha ** decay
      classes = ['tree','water','building','ground','road']
      [tf.summary.scalar('%s_weight'%classes[x],alpha[x]) for x in range(alpha.shape[-1].value)]
      weights = tf.reduce_sum(oh_lab * alpha,-1)

    with tf.variable_scope("FocalLoss") as scope:

      p_t     = tf.where(tf.equal(oh_lab,1),logits,1-logits)
      # Get Focal Loss
      focal_l = ( (1 - p_t)  ** tuning ) * ( - (tf.log(p_t+eps) ) )
      # Get Alpha Weighted Focal Loss
      focal_l = tf.reduce_sum(focal_l,-1) * weights
      # Reduce that shiiiiiiz
      focal_l = tf.reduce_mean(focal_l)

    tf.summary.scalar('Focal_Loss',focal_l)

    return focal_l


# Loss function for tape, using cross entropy
def weighted_xentropy_loss(labels,logits):

    weights = alpha_weight_gen_v2(labels)

    with tf.variable_scope("CrossEntropy") as scope:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(labels,tf.int32),logits = logits)

    loss = tf.cast(loss,tf.float32) * tf.cast(weights,tf.float32)
    loss = tf.reduce_mean(loss)

    tf.summary.scalar('XEnt_Loss',loss)
    loss = l2loss(loss)
    return loss

def alpha_weight_gen(labels):
  with tf.variable_scope("AlphaGeneration") as scope:
    oh_lab  = tf.one_hot(labels,FLAGS.num_classes)
    eps     = 1e-10
    alpha   = [1 / (tf.reduce_mean(oh_lab[:,:,:,x]) * FLAGS.num_classes + eps) for x in range(FLAGS.num_classes)]
    alpha   = tf.stack(alpha)

    step    = tf.train.get_global_step()
    decay   = tf.train.exponential_decay(.90,step,100,.99,staircase=True)
    tf.summary.scalar("AlphaDecay",decay)
    alpha   = alpha ** decay
    classes = ['tree','water','building','ground','road']
    [tf.summary.scalar('%s_weight'%classes[x],alpha[x]) for x in range(alpha.shape[-1].value)]
    weights = tf.reduce_sum(oh_lab * alpha,-1)
    return weights

def alpha_weight_gen_v2(labels):
  with tf.variable_scope("AlphaGeneration") as scope:
    im_size = FLAGS.imgW * FLAGS.imgH
    oh_lab  = tf.one_hot(labels,FLAGS.num_classes)
    eps     = 1e-10

    # This gives class frequencies
    alpha   = [tf.reduce_mean(oh_lab[:,:,:,x]) for x in range(FLAGS.num_classes)]
    alpha   = tf.stack(alpha)

    # This gives us inverse training weights. So, the more frequent a class is,
    # the less we want it to contribute to the loss. We never want it to be
    # actually 0, so we add a small epsilon.
    alpha   = 1-alpha + eps

    # As we train, we want these weights to approach 1, the way we do this is raise
    # these alpha weights to an exponentially decaying number. TF supplies this.
    step    = tf.train.get_global_step()
    decay   = tf.train.exponential_decay(.90,step,100,.99,staircase=True)

    # Report our decay.
    tf.summary.scalar("AlphaDecay",decay)

    # Raise alpha weights to the decay
    alpha   = alpha ** decay

    # Report the class weights
    classes = ['tree','water','building','ground','road']
    [tf.summary.scalar('%s_weight'%classes[x],alpha[x]) for x in range(alpha.shape[-1].value)]

    # Generate a full H,W loss multiplier for each pixel in a training example.
    weights = tf.reduce_sum(oh_lab * alpha,-1)
    return weights


# Absolute accuracy calculation for counting
def accuracy(labels_flat,logits_flat,name = 'Accuracy'):
  with tf.variable_scope(name) as scope:
    accuracy = tf.metrics.accuracy(labels = labels_flat, predictions = logits_flat)
    acc,up = accuracy
    tf.summary.scalar('Accuracy',tf.multiply(acc,100))
    return accuracy

# Absolute accuracy calculation for counting
def precision(labels_flat,logits_flat,name = 'Precision'):
  with tf.variable_scope(name) as scope:
    precision = tf.metrics.precision(labels = labels_flat, predictions = logits_flat)
    prec,up = precision
    tf.summary.scalar('Precision',prec)
    return accuracy

# Absolute accuracy calculation for counting
def recall(labels_flat,logits_flat,name = 'Recall'):
  with tf.variable_scope(name) as scope:
    recall = tf.metrics.recall(labels = labels_flat, predictions = logits_flat)
    rec,up = recall
    tf.summary.scalar('Recall',rec)
    return accuracy

def miou(labels,logits):
  with tf.variable_scope("MIOU") as scope:
    miou      = tf.metrics.mean_iou(labels = labels, predictions = logits, num_classes = FLAGS.num_classes)
    _miou,op  = miou
    tf.summary.scalar('MIOU',_miou)
    return miou

# Relative error calculation for counting
# |Lab - Log| / Possible Classes
def count_rel_err(labels,logits):
  with tf.variable_scope("rel_err_calc") as scope:
    err    = tf.subtract(labels,logits)
    err    = tf.abs(err)
    rel_err= tf.divide(err,labels)
    rel_err= tf.minimum(1.0,rel_err)
    rel_err= tf.squeeze(rel_err)
    if(FLAGS.batch_size > 1):
      rel_err   = tf.reduce_mean(rel_err,-1)

  tf.summary.scalar('Relative_Error',rel_err * 100)

  return rel_err

def count_huber_loss(labels,logits):
  with tf.variable_scope("Huber_Loss") as scope:
    huber = tf.losses.huber_loss(labels, logits)
  tf.summary.scalar("Huber_Loss",huber)
  return huber

def count_rmse(labels,logits,name = "RSME_Loss"):
  with tf.variable_scope(name) as scope:
    rmse = tf.reduce_mean(tf.sqrt(1 / FLAGS.batch_size * ((logits - labels) ** 2)))
  tf.summary.scalar(name,rmse)
  return rmse

def count_rel_rmse(labels,logits):
  with tf.variable_scope("Rel_RMSE_Loss") as scope:
    rel_rmse = tf.sqrt(1 / FLAGS.batch_size * tf.reduce_sum(((logits - labels) ** 2)/(labels + 1)))
  tf.summary.scalar("Rel_RMSE",rel_rmse)
  return rel_rmse

# Relative Accuracy calculation, I don't like this!
# |Lab - Log| / Lab
def count_rel_acc(labels,logits):
  with tf.variable_scope("rel_acc_calc") as scope:
    labels = tf.cast(labels,tf.float32)
    logits = tf.cast(logits,tf.float32)
    sum_acc= tf.Variable(0,dtype = tf.float32,name = 'Sum_Rel_Acc')

    l_min = tf.minimum(labels,logits)
    l_max = tf.maximum(labels,logits)
    rel = tf.divide(tf.abs(tf.subtract(logits,labels)),labels)

    zero = tf.constant(0,dtype=tf.float32,name='zero')
    one  = tf.constant(1,dtype=tf.float32,name='one')
    full_zeros = tf.zeros((FLAGS.batch_size),tf.float32)
    full_zeros = tf.squeeze(full_zeros)
    full_ones  = tf.ones((FLAGS.batch_size),tf.float32)
    full_ones  = tf.squeeze(full_ones)
    full_false = []
    for x in range(FLAGS.batch_size):
      full_false.append(False)
    full_false = tf.convert_to_tensor(full_false)
    full_false = tf.squeeze(full_false)

    # Get where NAN, inf, and logits is zero
    nans  = tf.is_nan(rel)
    infs  = tf.is_inf(rel)
    zeros = tf.equal(logits,zero)

    #If its a zero, get the NaN position, otherwise false.
    z_nan = tf.where(zeros,nans,full_false)
    z_inf = tf.where(zeros,infs,full_false)
    # Set to 1
    rel   = tf.where(z_nan,full_ones,rel)
    rel   = tf.where(z_inf,full_ones,rel)

    # Any leftover NaN or inf is where we counted wrong, so the rel acc is zero.
    nans  = tf.is_nan(rel)
    infs  = tf.is_inf(rel)
    rel   = tf.where(nans,full_zeros,rel)
    rel   = tf.where(infs,full_zeros,rel)

    # Get the minimum of relative acc or 1/ rel acc
    rel   = tf.minimum(rel,tf.divide(one,rel))

    rel = tf.reduce_mean(rel,-1)

  tf.summary.scalar('Relative Accuracy',tf.multiply(rel,100))
  return rel


def dense_reduction(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
                activation = lrelu, trainable = True,name = 'Dense_Block'):
  with tf.variable_scope(name) as scope:
    net = delist(net)
    for n in range(kmap):
      out = bn_conv2d(net, training, filters=filters, kernel=kernel, stride=1,
                        activation=activation, trainable=trainable, name = '_map_%d'%n)
      net = tf.concat([net,out],-1,name = '%d_concat'%n)
    if stride is not 1:
      net = max_pool(net,stride,stride)
    return net

  return net

def MSE(labels,logits,name = 'MSE'):
  with tf.variable_scope(name) as scope:
    return tf.reduce_mean(tf.pow(labels-logits,2))

def PNSR(labels,logits,max=255,name = 'PNSR'):
  with tf.variable_scope(name) as scope:
    val = 10 * tf.log(255**2 / MSE(labels,logits)) / tf.log(10.0)
  tf.summary.scalar(name,val)
  return val

def SSIM(labels,logits,max=255,name = 'SSIM'):
  with tf.variable_scope(name) as scope:
    x,y = labels,logits
    l = max
    k1,k2 = .01,.03
    c1,c2 = (k1*l)**2,(k2*l)**2
    tf.summary.scalar("c1",tf.reduce_mean(c1))
    tf.summary.scalar("c2",tf.reduce_mean(c2))
    mu_x,sig_x = tf.nn.moments(x,[1,2,3])
    tf.summary.scalar("Mu_X",tf.reduce_mean(mu_x))
    tf.summary.scalar("Sig_X",tf.reduce_mean(sig_x))
    mu_y,sig_y = tf.nn.moments(y,[1,2,3])
    tf.summary.scalar("Mu_X",tf.reduce_mean(mu_y))
    tf.summary.scalar("Sig_X",tf.reduce_mean(sig_y))
    bat,hei,wid,cha = x.get_shape().as_list()
    sig_xy = [tf.contrib.metrics.streaming_covariance(x[b],y[b])[0] for b in range(bat)]
    sig_xy = tf.stack(sig_xy)
    [tf.summary.scalar("Sig_XY",sig_xy[n]) for n in range(FLAGS.batch_size)]

    num = (2*mu_x*mu_y)            * (2*sig_xy + c2)
    dem = (mu_x**2 + mu_y**2 + c1) * (sig_x**2 + sig_y**2 + c2)

    val = num / dem

    val = tf.reduce_mean(val,-1)
  tf.summary.scalar(name,val)
  return val




''' PROVIDED BY https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py'''
def charbonnier_loss(labels, logits, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001, name = "Charbonnier"):
    x = labels - logits
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.
    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    with tf.variable_scope('charbonnier_loss'):
        batch, height, width, channels = tf.unstack(tf.shape(x))
        normalization = tf.cast(batch * height * width * channels, tf.float32)

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:
            error = tf.multiply(mask, error)

        if truncate is not None:
            error = tf.minimum(error, truncate)
        err = tf.reduce_sum(error) / normalization
    tf.summary.scalar(name,err)
    return err
