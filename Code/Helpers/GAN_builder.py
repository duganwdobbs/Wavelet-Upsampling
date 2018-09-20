import tensorflow as tf
from Helpers import ops, util, modules

flags = tf.app.flags
FLAGS = flags.FLAGS

# A class file to wrap a GAN.

# A GAN init gets:
#   - Arch for Generator
#   - Arch for Discriminator

# A GAN has:
#   - A Generator
#   - A Discriminator
#   - Names
#   - Scopes
#   - Reuse variables

# A GAN call gets:
#   - Sample Space Input
#   - Real Sample Output

# A GAN returns:
#   - Fake Production
#   - GAN loss
#   - Discriminator Loss

class GAN:
  def __init__(self,out_features,disc_size,name,gen_arch=None,disc_arch=None,resize_method = 'BILINEAR'):
    self.name         = name;
    # If it has run, we don't want new values, we want to resuse.
    self.d_reuse        = False
    self.g_reuse        = False
    # Set Generator Object
    self.gen_name     = name + '_Generator'
    self.disc_name    = name + '_Discriminator'
    self.out_features = out_features
    # This should be an image size in (H, W)
    self.disc_size    = disc_size
    self.resize_method = resize_method

  def __call__(self,z_input,real,run_disc):
    z_input = ops.delist(z_input)
    fake               = self.generator(z_input)
    gen_loss,disc_loss = self.discriminator(real,fake) if run_disc else (0,0)
    with tf.variable_scope(self.name) as scope:
      tf.summary.image("FAKE",fake)
      tf.summary.image("REAL",real)
    returns            = {"fake":fake,"g_loss":gen_loss,"d_loss":disc_loss}
    return returns

  def generator(self,z_input):
    with tf.variable_scope(self.name, reuse = self.g_reuse) as scope:
      self.g_reuse = True
      return self.Simple_Generator(z_input,self.gen_name)

  def discriminator(self,real,fake):
    with tf.variable_scope(self.name, reuse = self.d_reuse) as scope:
      self.d_reuse = True
      return self.Simple_Discriminator_Builder(real,fake,self.disc_name)

  def Simple_Generator(self,net,name):
    with tf.variable_scope(name) as scope:
      modules.Encoder_Decoder(net,3)
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
      noise_var = .85
      eps = 1e-4
      logs = logs * noise_var + tf.random_uniform(shape = logs.get_shape(),minval = eps,maxval = (1-noise_var))
      disc_loss = -tf.reduce_mean(tf.log(logs[0]) + tf.log(1 - logs[1]))
      gen_loss  = -tf.reduce_mean(tf.log(logs[1]))
    tf.summary.scalar('disc_loss',disc_loss)
    tf.summary.scalar('gen_loss',gen_loss)
    return disc_loss,gen_loss

  def Simple_Discriminator_Builder(self,real,fake,name):
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
      h,w = self.disc_size
      disc_imgs = tf.reshape(disc_imgs,(FLAGS.batch_size*2,h,w,3))
      disc_imgs = disc_imgs / (tf.reduce_max(disc_imgs)/2)-1
      disc_logs = self.Simple_Discriminator(disc_imgs,name=name)
      b,c       = disc_logs.get_shape().as_list()

      disc_real_logs = disc_logs[0   :b//2]
      disc_fake_logs = disc_logs[b//2:    ]

      disc_logs = tf.stack([disc_real_logs,disc_fake_logs],-1)
      # Shape objects to [B,H,W,C,Real/Fake]
      disc_logs = tf.transpose(disc_logs,(2,0,1))
      # Shape objects to [Real/Fake,B,H,W,C]
      return self.Discriminator_Loss(disc_logs,name)

  def Simple_Discriminator(self,imgs,name = None):
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
      print(h,w)
      net = imgs
      strides = util.factors(h,w)
      for x in range(len(strides[:-1])):
        stride = strides[x]
        input(stride)
        # Discard the skip connnection as we are just using the downsampled data
        _,net = modules.Encoder(net,2+x,3,stride,x)
      net = ops.conv2d(net,16,3)
      net = tf.reshape(net,(FLAGS.batch_size*2,-1))
      # Net will now be a B,#
      net = tf.layers.dense(net,100,activation=ops.relu)
      # Logit will be
      net = tf.layers.dense(net,1,activation = tf.nn.sigmoid)
      return net
