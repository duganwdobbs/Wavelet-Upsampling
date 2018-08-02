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
        self.generator.start(5)
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

   def Dense_Add_Block(self,net,level,features = None, kernel = 3, kmap = 4):
    with tf.variable_scope("Dense_Block_%d"%level) as scope:
      # If net is multiple tensors, concat them.
      net = ops.delist(net)
      if features is None:
        b,h,w,c = net.get_shape().as_list()
        features = c
      # Setup a list to store map outputs.
      outs = []

      # Dummy variable for training and trainable atm self.
      training = True
      trainable = True

      for n in range(kmap):
        # BN > RELU > CONV > DROPOUT, as per 100 Layers Tiramisu
        out  = ops.batch_norm(ops.delist([net,ops.delist(outs)]),training,trainable) if n > 0 else ops.batch_norm(net,training,trainable)
        out  = ops.relu(out)
        out  = tf.layers.dropout(out,FLAGS.keep_prob)
        out  = ops.conv2d(out,filters=features,kernel=kernel,stride=1,activation=None,padding='SAME',name = '_map_%d'%n)
        out  = tf.add_n([net,out])
        for n in outs:
          out = tf.add_n([out,n])
        outs.append(out)

      return outs[-1]


  '''-------------------------END HELPER FUNCTIONS----------------------------'''

  def encoder_decoder(self,net,out_features,name="Encoder_Decoder"):
    with tf.variable_scope(name) as scope:
      trainable   = True
      kmaps       = [ 2, 3]
      features    = [ 4, 6]
      strides     = [ 2, 3]
      skips       = []

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
      net = ops.conv2d(net,out_features,3,name='Decomp_Formatter',activation = None)
      return net

  def summary_image(self,name,img):
    with tf.variable_scope(name) as scope:
      img = img + tf.minimum(0.0,tf.reduce_min(img))
    tf.summary.image(name,img)

  def summary_wavelet(self,name,dwt_output):
    with tf.variable_scope(name) as scope:
      avg    = dwt_output[0,0,:,:,:,:]
      self.summary_image(name + "_avg",avg)
      low_w  = dwt_output[0,1,:,:,:,:]
      self.summary_image(name + "_low_w",low_w)
      low_h  = dwt_output[1,0,:,:,:,:]
      self.summary_image(name + "_low_h",low_h)
      detail = dwt_output[1,1,:,:,:,:]
      self.summary_image(name + "_detail",detail)

  def gen_aerr(self,labels,logits):
    with tf.variable_scope("AbsErrGen") as scope:
      err = tf.abs(labels-logits)
      err = tf.reduce_mean(err,-1)
      err = tf.expand_dims(err,-1)
      return err

  def inference(self):
    pywt_wavelet = "db2"
    wavelet = eval("wavelets." + pywt_wavelet)

    # Resize and norm images
    self.re_img = self.imgs[:,::2,::2,:]

    with tf.variable_scope("DWT") as scope:
      gt_dwt = wavelets.dwt(self.imgs, wavelet)
      self.gt_avg    = gt_dwt[0,0,:,:,:,:]
      self.gt_low_w  = gt_dwt[0,1,:,:,:,:]
      self.gt_low_h  = gt_dwt[1,0,:,:,:,:]
      self.gt_detail = gt_dwt[1,1,:,:,:,:]

    # Average Decomposition, what we're given
    # avg_scale = tf.Variable(2, name = "avg_scale",dtype = tf.float32)
    self.pred_avg    = self.encoder_decoder(self.re_img,3,"Avg_Gen")
    # Low pass Width
    self.pred_low_w  = self.encoder_decoder(self.re_img,3,"Low_w_Gen")
    # Low pass Height
    self.pred_low_h  = self.encoder_decoder(self.re_img,3,"Low_h_Gen")
    # High Pass
    self.pred_detail = self.encoder_decoder(self.re_img,3,"Detail_Gen")

    pred_dwt = tf.stack(
    [ tf.stack([self.pred_avg   , self.pred_low_w ],-1),
      tf.stack([self.pred_low_h , self.pred_detail],-1) ]
            ,-1)
    pred_dwt = tf.transpose(pred_dwt, [4,5,0,1,2,3])

    with tf.variable_scope("IDWT") as scope:
      self.w_x, self.wav_logs = wavelets.idwt(pred_dwt, wavelet)
    self.wav_logs = ops.relu(self.wav_logs)

    self.summary_wavelet("3_Pred_Wav",pred_dwt)
    self.summary_wavelet("3_GT_Wav",gt_dwt)
    self.summary_wavelet("4_Error_Wav",self.gen_aerr(gt_dwt,pred_dwt))

    self.summary_image("1_Origional"  ,self.imgs  )
    self.summary_image("1_Resized"    ,self.re_img)
    self.summary_image("1_WavResult"  ,self.wav_logs  )
    self.summary_image("2_Full_Error" ,self.gen_aerr(self.wav_logs,self.imgs))

    self.da_logs = self.Dense_Add_Block(self.wav_logs   ,1)
    self.da_logs = self.Dense_Add_Block(self.da_logs,2)
    self.da_logs = ops.relu(self.da_logs)

  # END INFERENCE

  def build_metrics(self):
    labels = self.imgs
    wav_rmse      = ops.count_rmse(labels,self.wav_logs,name = "Wav_RMSE")
    da_rmse       = ops.count_rmse(labels,self.da_logs,name = "Da_RMSE")

    wav_char      = ops.charbonnier_loss(labels,self.wav_logs,name = "Wav_Char")
    da_char       = ops.charbonnier_loss(labels,self.da_logs,name = "Da_Char")

    # Not enabling L2 loss, as counting networks might not work well with it.
    # total_loss = l2loss(huber,l2 = True)

    self.train = self.optomize(wav_char + da_char)
    self.metrics = {"Wav_RMSE":wav_rmse,"DA_RMSE":da_rmse,"Wav_Char":wav_char,"DA_Char":da_char}


  def optomize(self,loss,learning_rate = .001):
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
      self.losses.append(_metrics['DA_Char'])

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
