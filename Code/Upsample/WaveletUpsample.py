import tensorflow as tf, numpy as np, shutil, wget, tarfile, os
from Helpers     import ops, util
from DataLoaders import GoogleWebGetter as Generator

from Helpers.UTIL_wavelets.tfwt import TFWAV
from Helpers.GAN_builder import GAN

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
      self.sum_d_loss = []
      self.sum_g_loss = []
      self.sum_t_loss = []

      with tf.device('/gpu:1'):
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
    self.imgs      = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.imgH,FLAGS.imgW,3])

  # END INPUTS

  def summary_image(self,name,img):
    with tf.variable_scope(name) as scope:
      min = tf.reduce_min(img,(1,2,3))
      min = tf.reshape(min,[FLAGS.batch_size,1,1,1])
      img = img - min
    tf.summary.image(name,img)

  def summary_wavelet(self,name,ll,lh,hl,hh):
    with tf.variable_scope(name) as scope:
      self.summary_image(name + "_ll",ll)
      self.summary_image(name + "_lh",lh)
      self.summary_image(name + "_hl",hl)
      self.summary_image(name + "_hh",hh)

  def gen_aerr(self,labels,logits):
    with tf.variable_scope("AbsErrGen") as scope:
      err = tf.abs(labels-logits)
      err = tf.reduce_mean(err,-1)
      err = tf.expand_dims(err,-1)
      return err

  def Level_Error_Builder(self,labels,logits):
    with tf.variable_scope("Level_Error_Gen") as scope:
      rmse = ops.count_rmse(labels,logits,name = "RMSE")
      char = ops.charbonnier_loss(labels,logits,name = "CHAR")
      self.sum_t_loss.append(char)

  '''-------------------------END HELPER FUNCTIONS----------------------------'''

  def level_builder(self,level,gt,sc_img,ll = None,bottom = False,top = False):
    with tf.variable_scope("Level_%d"%level) as scope:
      gt_ll,gt_lh,gt_hl,gt_hh = self.wavelet.dwt(gt)

      # If we're at the bottom, we need to create our ll approximation
      lg_ll,g_loss,d_loss = self.ll_GAN([sc_img],gt_ll,True) if bottom else ll
      self.sum_g_loss.append(g_loss)
      self.sum_d_loss.append(d_loss)

      # High pass in the height
      lg_lh,g_loss,d_loss = self.lh_GAN([sc_img],gt_lh,True)
      self.sum_g_loss.append(g_loss)
      self.sum_d_loss.append(d_loss)

      # High pass in the width
      lg_hl,g_loss,d_loss = self.hl_GAN([sc_img],gt_hl,True)
      self.sum_g_loss.append(g_loss)
      self.sum_d_loss.append(d_loss)

      # The HH features can be considered a combination of the other detail
      #   channels, so the generator should be able to see them.
      lg_hh,g_loss,d_loss = self.hh_GAN([sc_img],gt_hh,True)
      self.sum_g_loss.append(g_loss)
      self.sum_d_loss.append(d_loss)

      w_x,result = self.wavelet.idwt(lg_ll,lg_lh,lg_hl,lg_hh)

      self.summary_wavelet('GT_WAV',gt_ll,gt_lh,gt_hl,gt_hh)
      self.summary_wavelet('LG_WAV',lg_ll,lg_lh,lg_hl,lg_hh)

      return result

  def pad(self,img):
    with tf.variable_scope("pad") as scope:
      pad_pixels  = 3
      pad_size    = (2 * 2 * 3) * pad_pixels
      paddings    = [[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]]
      img         = tf.pad(img,paddings,"REFLECT") / 255.0
      return img

  def depad(self,img,stride = 1):
    with tf.variable_scope("depad") as scope:
      pad_pixels  = 3
      pad_size    = (2 * 2 * 3) * pad_pixels // stride
      paddings    = [[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]]
      return img[:,pad_size:-pad_size,pad_size:-pad_size,:]

  def inference(self):
    # This is our wavelet.
    self.wavelet = TFWAV(FLAGS.wavelet_type,FLAGS.wavelet_train)

    img_s0      = self.pad(self.imgs)
    b,h,w,c     = img_s0.get_shape().as_list()

    # Setting up GANs
    disc_size    = (h//2,w//2)
    self.hh_GAN  = GAN(3,disc_size,"hh")
    self.hl_GAN  = GAN(3,disc_size,"hl")
    self.lh_GAN  = GAN(3,disc_size,"lh")
    self.ll_GAN  = GAN(3,disc_size,"ll")

    with tf.variable_scope("Downsample") as scope:
      img_s1 = img_s0[:,::2,::2,:]
      tf.summary.image("Downsamp",img_s1)

    # This wavelet upsample generates the full sized image using the source
    #   image and the generated ll feature from the previous level.
    self.logs = self.level_builder( 0 , gt = img_s0 , sc_img = img_s1 , bottom = True )

    self.logs = self.depad(self.logs)

    # Clip the values below zero
    # self.logs = tf.maximum( self.logs , 0.0   )
    # Clip the values above max
    # self.logs = tf.minimum( self.logs , 1.0   )

    self.Level_Error_Builder(self.imgs,self.logs * 255)

    # Undo our paddings
    tf.summary.image( "Origional" , self.imgs )
    tf.summary.image( "Result"    , self.logs )

  # END INFERENCE

  def build_metrics(self):
    labels = self.imgs
    logits = self.logs

    # Sum up the losses
    with tf.variable_scope("Loss_Summation") as scope:
      self.sum_d_loss = tf.reduce_sum(self.sum_d_loss)
      self.sum_g_loss = tf.reduce_sum(self.sum_g_loss)
      self.sum_t_loss = tf.reduce_sum(self.sum_t_loss)

    self.metrics = {"Wav_Char":self.sum_t_loss}

    disc_vars = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
    gen_vars  = [var for var in tf.trainable_variables() if 'Generator'     in var.name]

    disc_l2 = ops.l2loss(loss_vars = disc_vars, l2=True)
    gen_l2  = ops.l2loss(loss_vars = gen_vars , l2=True)

    total_disc_loss = self.sum_d_loss                         + disc_l2
    total_gen_loss  = self.sum_g_loss / 2.0 + self.sum_t_loss + gen_l2

    with tf.variable_scope("Metrics") as scope:
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
                                        decay_steps   = 1500,
                                        decay_rate    = .9,
                                        staircase     = False,
                                        name          = None
                                       )
    self.train = (self.optomize(total_gen_loss,gen_vars,self.global_step,learning_rate = gen_lr),self.optomize(total_disc_loss,disc_vars,learning_rate = FLAGS.learning_rate / 10))
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
