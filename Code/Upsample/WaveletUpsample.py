#SegmentationNetwork

import tensorflow as tf, numpy as np, tensorflow as tf, shutil, wget, tarfile, os
from Helpers     import ops, util
from DataLoaders import GoogleWebGetter as Generator

from Helpers.UTIL_wavelets.tfwt import TFWAV

# Import Flag globals
flags = tf.app.flags
FLAGS = flags.FLAGS

# Network Class File

# A network has:
# Session / Graph
# Inputs
# Inferece
# Logits
# Metrics
# Trainer
# Saver
# Runner
# MORE SUMMARIES
# DIRECTORIES

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
        if FLAGS.disc_type == 'image':
          self.Image_GAN_builder()
        if FLAGS.disc_type == 'wavelet':
          self.Wavelet_GAN_Builder()
        print("\rSETTING UP %s METRICS                   "%self.net_name,end='')
        self.build_metrics()

      print("\rINITIALIZING %s NETWORK                   "%self.net_name,end='')
      self.sess.run(tf.local_variables_initializer())
      self.sess.run(tf.global_variables_initializer())
      print("\rSETTING UP %s SAVER                       "%self.net_name,end='')
      self.build_saver()

      self.losses     = []
      self.net_losses = []
      self.logging_ids= []

      if not self.training:
        self.save()
      # If you want a copy of the code in your epoch folders, uncomment and change directory for this next line.
      # shutil.copytree(FLAGS.base_dir + '../Binalab-Seagrass/Code/DuganNetV6',self.log_dir+'bkupcode/')
      self.step = tf.train.global_step(self.sess,self.global_step) if self.training else 0
      print("\rNETWORK CLASS %s SETUP... PROCEEDING                       "%self.net_name,end='')
      if not restore:
        util.get_params()
  # END __init__

  def inputs(self):
    self.imgs      = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.imgH,FLAGS.imgW,3]) / 255
  # END INPUTS

  '''-----------------------------HELPER FUNCTIONS----------------------------'''

  def Dense_Block(self,net,level,features = 12, kernel = 3, kmap = 5, bn = True):
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

        if bn:
          out  = ops.batch_norm(ops.delist([net,outs]),training,trainable) if n > 0 else ops.batch_norm(net,training,trainable)
        else:
          out  = ops.delist([net,outs]) if n > 0 else net

        out  = ops.conv2d(out,filters=features,kernel=kernel,stride=1,activation=None,padding='SAME',name = '_map_%d'%n)
        out  = tf.nn.leaky_relu(out)
        out  = tf.layers.dropout(out,FLAGS.keep_prob)
        outs = tf.concat([outs,out],-1,name='%d_cat'%n) if n > 0 else out

      return outs

  def Transition_Down(self,net,stride,level):
    with tf.variable_scope("Transition_Down_%d"%level) as scope:
      net  = ops.delist(net)
      # Differentiation from 100 Layers: No 1x1 convolution here.
      # net  = tf.layers.dropout(net,FLAGS.keep_prob)
      net  = ops.avg_pool(net,stride,stride)
      return net

  def Transition_Up(self,net,stride,level):
    with tf.variable_scope("Transition_Up_%d"%level) as scope:
      net = ops.deconvxy(net,stride)
      return net

  def Encoder(self,net,kmap,feature,stride,level,bn = True):
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
      skip  = self.Dense_Block(net,level,features = feature,kmap = kmap,bn = bn)
      down  = self.Transition_Down([net,skip],stride,level)

    return skip,down

  def Decoder(self,net,skip,kmap,feature,stride,level,bn = True):
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

    Returns:
      net           : The output hidden network state
    '''
    with tf.variable_scope("Decoder_%d"%level) as scope:
      net   = self.Transition_Up(net,stride,level)
      resid = self.Dense_Block([net,skip],level,features = feature,kmap = kmap,bn = bn)
      net   = ops.delist([net,resid])

    return net

  def Encoder_Decoder(self,net,out_features,name="Encoder_Decoder",bn = True):
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

        skip,net = self.Encoder(net,kmap,feature,stride,x+1,bn)

        skips.append(skip)

      for x in range(1,len(strides)+1):
        skip   = skips[-x]
        stride = strides[-x]
        kmap   = kmaps[-x]
        feature= features[-x]

        net = self.Decoder(net,skip,kmap,feature,stride,len(strides)+1-x,bn)
      net = ops.conv2d(net,out_features,3,name='Decomp_Formatter',activation = None)
      return net

  def Simple_Wavelet_Generator(self,net,out_features,name = 'Simple_Wavelet_Generator'):
    with tf.variable_scope(name) as scope:
      net = ops.conv2d(net, 8,5,stride=1,activation=tf.nn.crelu,name='conv1')
      net = ops.conv2d(net,16,5,stride=1,activation=tf.nn.crelu,name='conv2')
      net = ops.conv2d(net,24,3,stride=1,activation=tf.nn.crelu,name='conv3')
      net = ops.conv2d(net,32,3,stride=1,activation=tf.nn.crelu,name='conv4')

      net = ops.conv2d(net,out_features,3,stride=1,activation=None,name='convEnd')
      return net

  def Discriminator(self,imgs,name = None):
    '''
    This is a standard Discriminator network that uses dense blocks followed by
      pooling in order to generate a logit to see if the image is real or fake.
    Parameters:
      imgs          : The images to determine if real or fake.
      name          : This The name of the Discriminator

    Returns:
      net           : The logits for the current images. (Value between 0-1)
    '''
    with tf.variable_scope(name) as scope:
      b,h,w,c = imgs.get_shape().as_list()
      net = imgs
      strides = util.factors(h,w)
      for x in range(len(strides[:-1])):
        stride = strides[x]
        # Discard the skip connnection as we are just using the downsampled data
        _,net = self.Encoder(net,2+x,3,stride,x)
      net = ops.conv2d(net,16,3)
      net = tf.reshape(net,(FLAGS.batch_size*2,-1))
      # Net will now be a B,#
      net = tf.layers.dense(net,100,activation=ops.relu)
      # Logit will be
      net = tf.layers.dense(net,1,activation = tf.nn.sigmoid)
      return net

  def Discriminator_Loss(self,logs,name):
    '''
    This method serves to generate GAN loss.
    Parameters:
      logs          : The logits to generate loss on. Shape [Real/Fake,B,H,W,C]
      name          : This The name of the module loss is being built for

    Returns:
      disc_loss     : The Discriminator loss to minimize
      gen_loss      : The loss to add to the generator
    '''
    with tf.variable_scope(name) as scope:
      noise_var = .8
      eps = 1e-5
      logs = logs * noise_var + tf.random_uniform(shape = logs.get_shape(),minval = eps,maxval = (1-noise_var))
      disc_loss = -tf.reduce_mean(tf.log(logs[0]) + tf.log(1 - logs[1]))
      gen_loss  = -tf.reduce_mean(tf.log(logs[1]))
    tf.summary.scalar('disc_loss',disc_loss)
    tf.summary.scalar('gen_loss',gen_loss)
    return disc_loss,gen_loss

  def Wavelet_Discriminator_Builder(self,real,fake,name):
    '''
    This method serves to build a Discriminator for real and fake logits
    Parameters:
      real          : The real logits
      fake          : The fake logits
      name          : The name of the Discriminator

    Returns:
      disc_loss     : The Discriminator loss to minimize
      gen_loss      : The loss to add to the generator
    '''
    with tf.variable_scope(name) as scope:
      disc_imgs = tf.concat([real,fake],0)
      b,h,w,c = self.pad_imgs.get_shape().as_list()
      h = h // 2
      w = w // 2
      disc_imgs = tf.reshape(disc_imgs,(b*2,h,w,c))
      disc_imgs = disc_imgs / (tf.reduce_max(disc_imgs)/2)-1
      disc_logs = self.Discriminator(disc_imgs,name=name)
      b,c       = disc_logs.get_shape().as_list()

      disc_real_logs = disc_logs[0   :b//2]
      disc_fake_logs = disc_logs[b//2:    ]

      disc_logs = tf.stack([disc_real_logs,disc_fake_logs],-1)
      # Shape objects to [B,H,W,C,Real/Fake]
      disc_logs = tf.transpose(disc_logs,(2,0,1))
      # Shape objects to [Real/Fake,B,H,W,C]
      return self.Discriminator_Loss(disc_logs,name)

  def Wavelet_GAN_Builder(self):
    real_wav = self.gt_dwt
    fake_wav = self.pred_dwt

    # disc_avg_loss,gen_avg_loss = self.Wavelet_Discriminator_Builder(real_wav[0,0],fake_wav[0,0],name='Avg_Discriminator')
    disc_avg_loss,gen_avg_loss = (0,0)
    disc_wid_loss,gen_wid_loss = self.Wavelet_Discriminator_Builder(real_wav[0,1],fake_wav[0,1],name='Wid_Discriminator')
    disc_hei_loss,gen_hei_loss = self.Wavelet_Discriminator_Builder(real_wav[1,0],fake_wav[1,0],name='Hei_Discriminator')
    disc_det_loss,gen_det_loss = self.Wavelet_Discriminator_Builder(real_wav[1,1],fake_wav[1,1],name='Det_Discriminator')

    with tf.variable_scope("Total_Gan_Losses") as scope:
      self.disc_loss = tf.reduce_mean([disc_avg_loss,disc_wid_loss,disc_hei_loss,disc_det_loss])
      tf.summary.scalar("Total_Disc_Loss",self.disc_loss)
      self.gen_loss  = tf.reduce_mean([gen_avg_loss ,gen_wid_loss, gen_hei_loss, gen_det_loss ])
      tf.summary.scalar("Total_Gen_Loss",self.gen_loss)

  def Image_Discriminator_builder(self,real,fake,name):
    '''
    This method serves to build a Discriminator for real and fake logits
    Parameters:
      real          : The real logits
      fake          : The fake logits
      name          : The name of the Discriminator

    Returns:
      disc_loss     : The Discriminator loss to minimize
      gen_loss      : The loss to add to the generator
    '''
    with tf.variable_scope(name) as scope:
      disc_imgs = tf.concat([real,fake],0)
      b,h,w,c   = self.imgs.get_shape().as_list()
      disc_imgs = tf.reshape(disc_imgs,(b*2,h,w,c))
      disc_logs = self.Discriminator(disc_imgs,name=name)
      b,c   = disc_logs.get_shape().as_list()
      disc_real_logs = disc_logs[0   :b//2]
      disc_fake_logs = disc_logs[b//2:    ]
      disc_logs = tf.stack([disc_real_logs,disc_fake_logs],-1)
      # Shape objects to [B,H,W,C,Real/Fake]
      disc_logs = tf.transpose(disc_logs,(2,0,1))
      # Shape objects to [Real/Fake,B,H,W,C]
      return self.Discriminator_Loss(disc_logs,name)

  def Image_GAN_builder(self):
    real_img = self.imgs
    fake_img = self.wav_logs

    self.disc_loss,self.gen_loss = self.Image_Discriminator_builder(real_img,fake_img,name='FullImgDiscriminator')

    tf.summary.scalar("Total_Disc_Loss",self.disc_loss)
    tf.summary.scalar("Total_Gen_Loss",self.gen_loss)

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

  '''-------------------------END HELPER FUNCTIONS----------------------------'''

  def inference(self):
    # This is our wavelet.
    wavelet = TFWAV(FLAGS.wavelet_train,FLAGS.wavelet_type)

    # Resize images and pad to deal with convolutional issues with wavelet
    #  transforms on borders.
    pad_pixels   = 3
    pad_size     = (2 * 2 * 3) * pad_pixels
    paddings     = [[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]]
    self.pad_imgs= tf.pad(self.imgs,paddings,"REFLECT")
    self.re_img  = self.pad_imgs[:,::2,::2,:]

    # Create Ground Truth wavelet features to compare against. Note: LOSS IS NOT
    #   CALCULATED WITH THESE
    with tf.variable_scope("DWT") as scope:
      self.gt_dwt = wavelet.dwt(self.pad_imgs)
      self.gt_avg    = self.gt_dwt[0,0]
      self.gt_low_w  = self.gt_dwt[0,1]
      self.gt_low_h  = self.gt_dwt[1,0]
      self.gt_detail = self.gt_dwt[1,1]

    # Create Generators for wavelet features
    # Average Decomposition, what we're given
    self.pred_avg    = self.Simple_Wavelet_Generator(self.re_img,3,"Avg_Generator")
    # Low pass Width
    self.pred_low_w  = self.Encoder_Decoder(self.re_img,3,"Low_w_Generator",False)
    # Low pass Height
    self.pred_low_h  = self.Encoder_Decoder(self.re_img,3,"Low_h_Generator",False)
    # High Pass
    self.pred_detail = self.Encoder_Decoder(self.re_img,3,"Detail_Generator",False)

    # Format our wavelet features for IDWT
    with tf.variable_scope('Wavelet_Formatting') as scope:
      pred_dwt = tf.stack(
      [ tf.stack([self.pred_avg   , self.pred_low_w ],-1),
        tf.stack([self.pred_low_h , self.pred_detail],-1) ]
              ,-1)
      self.pred_dwt = tf.transpose(pred_dwt, [4,5,0,1,2,3])

    # Preform our inverse discrete wavelet transform
    with tf.variable_scope("IDWT") as scope:
      self.w_x, self.wav_logs = wavelet.idwt(self.pred_dwt)

    # Clip the values below zero
    self.wav_logs = tf.maximum(self.wav_logs,0)
    # Clip the values above max
    self.wav_logs = tf.minimum(self.wav_logs,255)
    # Undo our paddings
    self.wav_logs = self.wav_logs[:,pad_size:-pad_size,pad_size:-pad_size,:]
    # Wavelet decom / recomp yields [?,?,?,?] shapes, we just need to reinforce
    #  static shapes in the code using this.
    self.wav_logs = tf.reshape(self.wav_logs,self.imgs.get_shape().as_list())

    # Log the goal image
    self.summary_image("1_Origional"  ,self.imgs  )
    # Log the predicted image
    self.summary_image("1_WavResult"  ,self.wav_logs  )
    # Log our resized image, this will have padding
    self.summary_image("2_Resized"    ,self.re_img)
    # Log our full sized image error as a viewable image.
    self.summary_image("2_Full_Error" ,self.gen_aerr(self.wav_logs,self.imgs))

    # Log our Pred wavelet features
    self.summary_wavelet("3_Pred_Wav",self.pred_dwt)
    # Log our GT wavelet features
    self.summary_wavelet("3_GT_Wav",self.gt_dwt)

    # For logging sake, generate error in wavelet features.
    with tf.variable_scope("Wav_Err_Gen") as scope:
      wav_err = self.gen_aerr(self.gt_dwt,self.pred_dwt)
      wav_avg_err = tf.reduce_mean(wav_err[0,0])
      wav_wid_err = tf.reduce_mean(wav_err[0,1])
      wav_hei_err = tf.reduce_mean(wav_err[1,0])
      wav_det_err = tf.reduce_mean(wav_err[1,1])
      tf.summary.scalar("wav_avg_err",wav_avg_err)
      tf.summary.scalar("wav_wid_err",wav_wid_err)
      tf.summary.scalar("wav_hei_err",wav_hei_err)
      tf.summary.scalar("wav_det_err",wav_det_err)

    self.summary_wavelet("4_Error_Wav",self.gen_aerr(self.gt_dwt,self.pred_dwt))
  # END INFERENCE

  def build_metrics(self):
    labels = self.imgs
    logits = self.wav_logs
    wav_rmse      = ops.count_rmse(labels,logits,name = "Wav_RMSE")
    wav_char      = ops.charbonnier_loss(labels,logits,name = "Wav_Char")

    # Not enabling L2 loss, as counting networks might not work well with it.
    # total_loss = l2loss(huber,l2 = True)

    self.metrics = {"Wav_RMSE":wav_rmse,"Wav_Char":wav_char}

    disc_vars = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
    gen_vars  = [var for var in tf.trainable_variables() if 'Generator'     in var.name]

    disc_l2 = ops.l2loss(loss_vars = disc_vars, l2=True)
    gen_l2  = ops.l2loss(loss_vars = gen_vars , l2=True)

    total_disc_loss = self.disc_loss + disc_l2
    total_gen_loss  = self.gen_loss  + wav_char

    PSNR = tf.image.psnr(labels, logits,255)
    tf.summary.scalar("PSNR",tf.reduce_mean(PSNR))
    SSIM = tf.image.ssim(labels, logits,255)
    tf.summary.scalar("SSIM",tf.reduce_mean(SSIM))
    SSIMM= tf.image.ssim_multiscale(labels, logits,255)
    tf.summary.scalar("SSIMM",tf.reduce_mean(SSIMM))

    with tf.variable_scope("PerImageStats") as scope:
      [tf.summary.scalar("PSNR_%d"%x,PSNR[x]) for x in range(FLAGS.batch_size)]
      [tf.summary.scalar("SSIM_%d"%x,SSIM[x]) for x in range(FLAGS.batch_size)]
      [tf.summary.scalar("SSIMM_%d"%x,SSIMM[x]) for x in range(FLAGS.batch_size)]

    gen_lr = tf.train.exponential_decay(
                                        learning_rate = FLAGS.learning_rate,
                                        global_step   = self.global_step,
                                        decay_steps   = 800,
                                        decay_rate    = .9,
                                        staircase     = False,
                                        name          = None
                                       )

    self.train = (self.optomize(total_gen_loss,gen_vars,self.global_step,learning_rate = gen_lr),self.optomize(total_disc_loss,disc_vars))
  # END BUILD_METRICS

  def optomize(self,loss,var_list = None,global_step = None,learning_rate = None):
    learning_rate = FLAGS.learning_rate if learning_rate is None else learning_rate
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # optomizer = tf.train.RMSPropOptimizer(learning_rate,decay = 0.9, momentum = 0.3)
      optomizer = tf.train.AdamOptimizer(learning_rate,epsilon = 1e-5)
      train     = optomizer.minimize(loss,global_step)
      return train
  # END OPTOMIZE

  def build_saver(self):
    if self.restore or not self.training:
      try:
        if not FLAGS.restore_disc:
          raise KeyError
        self.saver= tf.train.Saver()
        self.saver.restore(self.sess,tf.train.latest_checkpoint(FLAGS.run_dir,latest_filename = self.save_name))
      except:
        print('\rDISCRIMM VARS NOT LOADED... TRYING GEN ONLY!')
        gen_vars  = [var for var in tf.global_variables() if 'Discriminator' not in var.name and 'Adam' not in var.name]
        self.saver     = tf.train.Saver(gen_vars)
        self.saver.restore(self.sess,tf.train.latest_checkpoint(FLAGS.run_dir,latest_filename = self.save_name))
    self.saver     = tf.train.Saver()
    self.summaries = tf.summary.merge_all()
    self.writer    = tf.summary.FileWriter(self.log_dir,self.sess.graph)
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
    if FLAGS.num_steps is not None and self.step > FLAGS.num_steps:
      raise KeyError

    print('\rGetting next batch...',end='')
    _imgs,_ids = self.generator.get_next_batch(FLAGS.batch_size)
    print('\rGot next batch...',end='')

    fd               = {self.imgs : _imgs}

    _logs = 0

    # This block fires if we're testing.
    if not self.training:
      # Run the network
      _metrics,summaries = self.sess.run(test_op,feed_dict = fd)
      # If we're using advanced tensorboard logging, this runs.
      if FLAGS.adv_logging:
        self.writer.add_run_metadata(run_metadata,'step%d'%tf.train.global_step(self.sess,self.global_step))

      # Write the summaries to tensorboard
      self.writer.add_summary(summaries,tf.train.global_step(self.sess,self.global_step))

      self.logging_ids.append(_ids)
      self.losses.append(_metrics['DA_Char'])

    # This is a logging step.
    elif self.step % 10 == 0:
      _,summaries = self.sess.run(op,feed_dict = fd)
      print('\rSTEP %d'%self.step,end=' ')

      # If we're using advanced tensorboard logging, this runs.
      if FLAGS.adv_logging:
        self.writer.add_run_metadata(run_metadata,'step%d'%tf.train.global_step(self.sess,self.global_step))

      # Write the summaries to tensorboard
      self.writer.add_summary(summaries,tf.train.global_step(self.sess,self.global_step))

      # This is a model saving step.
      if self.step % 1000 == 0:
        self.save(self.global_step)

    # If we're not testing or logging, just train.
    else:
      _ = self.sess.run(t_op,feed_dict = fd)
    # END RUN

  def close(self):
    self.sess.close()
  # END CLOSE
