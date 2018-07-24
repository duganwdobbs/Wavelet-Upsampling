#SegmentationNetwork

import tensorflow as tf, numpy as np, tensorflow as tf, shutil, wget, tarfile, os
from Helpers     import ops, util
from DataLoaders import GoogleWebGetter as Generator

from Helpers.UTIL_wavelets import wavelets

# Import Flag globals
flags = tf.app.flags
FLAGS = flags.FLAGS

# Network Class File

# A network has:
# Session / Graph
# Inputs
# TODO: Inferece
# TODO: Logits
# TODO: Metrics
# TODO: Trainer
# Saver
# Runner
# TODO: MORE SUMMARIES
# TODO: DIRECTORIES

class ANN:
  def __init__(self,training,split,restore,timestr):
    with tf.Graph().as_default():
      self.net_name    = "WaveletUpsample"
      print("\rSETTING UP %s NETWORK"%self.net_name,end='                       ')
      config = tf.ConfigProto(allow_soft_placement = True)
      config.gpu_options.allow_growth = True
      self.sess        = tf.Session(config = config)

      self.global_step = tf.Variable(0,name='global_step',trainable = False)
      self.training    = training
      self.restore     = restore

      self.ckpt_name   = self.net_name + '.ckpt'
      self.save_name   = self.net_name + '.save'
      self.savestr     = FLAGS.run_dir + self.ckpt_name
      self.filestr     = FLAGS.run_dir + 'tensorlogs/' + timestr + '/' + self.net_name + '/'
      self.log_dir     = self.filestr
      # print("LOGGING TO  : %s. "%self.log_dir)

      with tf.device('/gpu:0'):
        ops.init_scope_vars()
        print("\rSETTING UP %s INPUTS                    "%self.net_name,end='')
        self.generator = Generator.DataGenerator(split,FLAGS.data_dir)
        self.inputs()
        print("\rSETTING UP %s INFERENCE                 "%self.net_name,end='')
        self.inference()
        print("\rSETTING UP %s METRICS                   "%self.net_name,end='')
        self.build_metrics()

      print("\rINITIALIZING %s NETWORK                   "%self.net_name,end='')
      self.sess.run(tf.local_variables_initializer())
      self.sess.run(tf.global_variables_initializer())
      print("\rSETTING UP %s SAVER                       "%self.net_name,end='')
      self.build_saver()

      self.step       = tf.train.global_step(self.sess,self.global_step) if self.training else 0
      self.losses     = []
      self.net_losses = []
      self.logging_ids= []

      if not self.training:
        self.save()
      # If you want a copy of the code in your epoch folders, uncomment and change directory for this next line.
      # shutil.copytree(FLAGS.base_dir + '../Binalab-Seagrass/Code/DuganNetV6',self.log_dir+'bkupcode/')
      print("\rNETWORK CLASS %s SETUP... PROCEEDING                       "%self.net_name,end='')
      if not restore:
        util.get_params()

  # END __init__

  def inputs(self):
    # TODO: Define Inputs
    self.imgs      = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.imgH,FLAGS.imgW,3]) / 255

  # END INPUTS

  '''-----------------------------HELPER FUNCTIONS----------------------------'''

  def Dense_Block(self,net,level,features = 12, kernel = 3, kmap = 5):
    with tf.variable_scope("Dense_Block_%d"%level) as scope:
      # If net is multiple tensors, concat them.
      net = ops.delist(net)
      # Setup a list to store map outputs.
      outs = []

      # Dummy variable for training and trainable atm self.
      training = True
      trainable = True

      for n in range(kmap):
        # BN > RELU > CONV > DROPOUT, as per 100 Layers Tiramisu
        out  = ops.batch_norm(ops.delist([net,outs]),training,trainable) if n > 0 else ops.batch_norm(net,training,trainable)
        out  = ops.relu(out)
        out  = ops.conv2d(out,filters=features,kernel=kernel,stride=1,activation=None,padding='SAME',name = '_map_%d'%n)
        out  = tf.layers.dropout(out,FLAGS.keep_prob)
        outs = tf.concat([outs,out],-1,name='%d_cat'%n) if n > 0 else out

      return outs

  def Transition_Down(self,net,stride,level):
    with tf.variable_scope("Transition_Down_%d"%level) as scope:
      net  = ops.delist(net)
      # Differentiation from 100 Layers: No 1x1 convolution here.
      net  = tf.layers.dropout(net,FLAGS.keep_prob)
      net  = ops.max_pool(net,stride,stride)
      return net

  def Transition_Up(self,net,stride,level):
    with tf.variable_scope("Transition_Up_%d"%level) as scope:
      net = ops.deconvxy(net,stride)
      return net

  def Encoder(self,net,kmap,feature,stride,level):
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
      skip  = self.Dense_Block(net,level,features = feature,kmap = kmap)
      down  = self.Transition_Down([net,skip],stride,level)

    return skip,down

  def Decoder(self,net,skip,kmap,feature,stride,level,residual_conn = True):
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
      net   = self.Transition_Up(net,stride,level)
      resid = self.Dense_Block([net,skip],level,features = feature,kmap = kmap)
      net   = ops.delist([net,resid])

    return net

  '''-------------------------END HELPER FUNCTIONS----------------------------'''

  def summary_image(self,name,img):
    tf.summary.image(name,img)

  def inference(self):
    trainable   = True
    kmaps       = [ 4, 6]
    features    = [ 6, 8]
    strides     = [ 2, 3]
    skips       = []

    # Resize and norm images
    self.re_img = self.imgs[:,::2,::2,:]
    net         = self.re_img

    for x in range(len(strides)):
      level = x+1
      stride = strides[x]
      kmap   = kmaps[x]
      feature= features[x]

      skip,net = self.Encoder(net,kmap,feature,stride,x+1)

      skips.append(skip)

    for x in range(1,len(strides)+1):
      skip   = skips[-x]
      stride = strides[-x]
      kmap   = kmaps[-x]
      feature= features[-x]

      net = self.Decoder(net,skip,kmap,feature,stride,len(strides)+1-x)

    decomps = ops.conv2d(net,9,3,name='Decomp_Formatter')
    to_idwt = tf.zeros([2,2,FLAGS.batch_size,FLAGS.imgH,FLAGS.imgW,3],tf.float32)

    # Average Decomposition, what we're given
    avg_scale = 1.41421356237 #tf.Variable(1.5, name = "avg_scale",dtype = tf.float32)
    tf.summary.scalar("Scale_Mul",avg_scale)
    self.re_img = self.re_img * avg_scale
    # Low pass Width
    self.low_w  = decomps[:,:,:,0:3]
    # Low pass Height
    self.low_h  = decomps[:,:,:,3:6]
    # High Pass
    self.high_p = decomps[:,:,:,6:9]

    dwt = tf.stack(
    [ tf.stack([self.re_img,self.low_w],-1),
      tf.stack([self.low_h,self.high_p],-1) ]
            ,-1)
    dwt = tf.transpose(dwt, [4,5,0,1,2,3])

    pywt_wavelet = "db2"
    wavelet = eval("wavelets." + pywt_wavelet)

    self.w_x, self.logs = wavelets.idwt(dwt, wavelet)
    # self.logs = self.logs + tf.reduce_min(self.logs)
    self.logs = ops.relu(self.logs)
    # self.logs = self.logs / tf.reduce_max(self.logs)

    self.abs_er = self.imgs - self.logs
    self.abs_er = tf.abs(self.abs_er / tf.reduce_max(self.abs_er))
    self.abs_er = tf.reduce_mean(self.abs_er,-1)
    b,h,w = self.abs_er.get_shape().as_list()
    self.abs_er = tf.reshape(self.abs_er,[b,h,w,1])

    self.summary_image("Average"  ,self.re_img)
    self.summary_image('Low_W'    ,self.low_w )
    self.summary_image('Low_H'    ,self.low_h )
    self.summary_image('High_'    ,self.high_p)
    self.summary_image("Origional",self.imgs  )
    self.summary_image("Result"   ,self.logs  )
    self.summary_image("Error"    ,self.abs_er)

  # END INFERENCE

  def build_metrics(self):
    labels,logits = self.imgs,self.logs
    rmse          = ops.count_rmse(labels,logits)

    # Not enabling L2 loss, as counting networks might not work well with it.
    # total_loss = l2loss(huber,l2 = True)

    self.train = self.optomize(rmse)
    self.metrics = {"RMSE":rmse}


  def optomize(self,loss,learning_rate = .0001):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # optomizer = tf.train.RMSPropOptimizer(learning_rate,decay = 0.9, momentum = 0.3)
      optomizer = tf.train.AdamOptimizer(learning_rate,epsilon = 1e-5)
      train     = optomizer.minimize(loss,self.global_step)
      return train
  # END OPTOMIZE

  def build_saver(self):
    self.saver     = tf.train.Saver()
    self.summaries = tf.summary.merge_all()
    self.writer    = tf.summary.FileWriter(self.log_dir,self.sess.graph)

    if self.restore or not self.training:
      self.saver.restore(self.sess,tf.train.latest_checkpoint(FLAGS.run_dir,latest_filename = self.save_name))
  # END SAVER

  def save(self,step=None):
    if self.training:
      self.saver.save(self.sess,self.savestr ,global_step = step,latest_filename = self.save_name)
    self.saver.save(self.sess,self.log_dir ,global_step = step,latest_filename = self.save_name)
  # END SAVE

  def run(self):
    t_op      = self.train
    op        = [self.train,self.summaries]
    test_op   = [self.metrics,self.summaries]
    self.step+= 1

    _imgs,_ids = self.generator.get_next_batch(FLAGS.batch_size)
    fd               = {self.imgs : _imgs}

    _logs = 0

    # This block fires if we're testing.
    if not self.training:
      # Run the network
      _metrics,summaries = self.sess.run(test_op,feed_dict = fd)
      # If we're using advanced tensorboard logging, this runs.
      if FLAGS.adv_logging:
        self.writer.add_run_metadata(run_metadata,'step%d'%step)

      # Write the summaries to tensorboard
      self.writer.add_summary(summaries,self.step)

      self.logging_ids.append(_ids)
      self.losses.append(metrics['Huber'])

    # This is a logging step.
    elif self.step % 10 == 0:
      _,summaries = self.sess.run(op,feed_dict = fd)
      print('\rSTEP %d'%self.step,end='')

      # If we're using advanced tensorboard logging, this runs.
      if FLAGS.adv_logging:
        self.writer.add_run_metadata(run_metadata,'step%d'%step)

      # Write the summaries to tensorboard
      self.writer.add_summary(summaries,self.step)

      # This is a model saving step.
      if self.step % 100 == 0:
        self.save(self.step)

    # If we're not testing or logging, just train.
    else:
      _ = self.sess.run(t_op,feed_dict = fd)
    # END RUN

  def close(self):
    self.sess.close()
  # END CLOSE
