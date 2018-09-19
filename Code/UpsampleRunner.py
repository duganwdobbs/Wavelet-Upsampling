import os,time,shutil,platform,contextlib,io
from multiprocessing import Process

import numpy         as     np
import tensorflow    as     tf

from Upsample.WaveletUpsample  import ANN


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS

if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'   , 'E:/Wavelet-Upsampling/'               , 'Base os specific DIR')
  flags.DEFINE_string ('code_dir'   , 'E:/Wavelet-Upsampling/Code/'          , 'Location of the code files.')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'   , '/data0/ddobbs/Wavelet-Upsampling/'    , 'Base os specific DIR')
  flags.DEFINE_string ('code_dir'   , '/data1/ddobbs/Wavelet-Upsampling/Code', 'Location of the code files.')

# Network Variables
flags.DEFINE_boolean('adv_logging'  , False                                  ,'If we log metadata and histograms.                                       DEFAULT = False'  )
flags.DEFINE_boolean('l2_loss'      , True                                   ,'If we use L2 loss.                                                       DEFAULT = True'   )
flags.DEFINE_boolean('restore'      , False                                  ,'If we use an old network state.                                          DEFAULT = False'  )
flags.DEFINE_boolean('restore_disc' , False                                  ,'If we restore discriminator vars.                                        DEFAULT = False'  )
flags.DEFINE_integer('num_epochs'   , 1                                      ,'Number of epochs to run per validation.                                  DEFAULT = 1'      )
flags.DEFINE_integer('num_steps'    , 24000                                  ,'Number of steps to train.                                                DEFAULT = None'   )
flags.DEFINE_integer('batch_size'   , 4                                      ,'Batch size for training.                                                 DEFAULT = ?'      )
flags.DEFINE_float  ('keep_prob'    , .9                                     ,'A variable to use for dropout percentage. (Dont dropout during testing!) DEFAULT = .9'     )
flags.DEFINE_float  ('learning_rate', .0001                                   ,'A variable to use for initial learning rate.                             DEFAULT = .001'   )
flags.DEFINE_string ('wavelet_type' , 'db2'                                  ,'The type of wavelet we use.                                              DEFAULT = \'DB2\'')
flags.DEFINE_boolean('wavelet_train', False                                  ,'If we are training wavelet vars.                                         DEFAULT = False'  )

# Directory and Checkpoint Information
flags.DEFINE_string ('run_dir'      , FLAGS.base_dir  + 'network_log/'       ,'Location to store the Tensorboard Output')
flags.DEFINE_string ('train_dir'    , FLAGS.base_dir                         ,'Location of the tfrecord files. Used if using TFRecord Generator')
flags.DEFINE_string ('data_dir'     , FLAGS.base_dir                         ,'Location of the training / testing / validation files. Used if using standard file generation.')
flags.DEFINE_string ('ckpt_name'    ,'WaveletUpsample.ckpt'                  ,'Checkpoint name')
flags.DEFINE_string ('ckpt_i_name'  ,'WaveletUpsample-interrupt.ckpt'        ,'Interrupt Checkpoint name')

# Helper function to start Tensorboard
def launchTensorBoard():
  with contextlib.redirect_stdout(io.StringIO()):
    os.system('tensorboard --logdir '+FLAGS.run_dir+'/tensorlogs/')

def train(train_run = True, restore = False,epoch = 0):
  if not train_run:
    FLAGS.batch_size = 1
    FLAGS.num_epochs = 4
    FLAGS.keep_prob  = 1
  else:
    FLAGS.batch_size = 3
    FLAGS.num_epochs = 1
    FLAGS.keep_prob  = .8

  split   = "TRAIN" if train_run else "TEST"
  timestr = split + '/' + time.strftime("%b_%Y_%H_%M",time.localtime()) + "_Epoch_%d"%epoch

  tf.reset_default_graph()
  net = ANN(train_run,split,restore,timestr)

  # Starts the input generator
  print("\rSTARTING INPUT GENERATION THREADS...                        ",end='')
  coord          = tf.train.Coordinator()
  threads        = tf.train.start_queue_runners(sess = net.sess, coord = coord)
  stfstr = 'TRAINING' if train_run else 'TESTING'
  print("\rSTARTING %s...                                        "%stfstr,end='')

  try:
    while not coord.should_stop():
      # Run the network and write summaries
      net.run()

  except KeyboardInterrupt:
    if train_run:
      net.save()
  except (tf.errors.OutOfRangeError,KeyError):
    print("\rDone.                                                            ")
  finally:
    if train_run:
      net.save()
    coord.request_stop()
  coord.join(threads)

  losses = 0
  met    = 0
  cmat   = 0
  if not train_run:
    losses = np.average(net.losses)
    cmat   = net.sum_cmat

  net.close()
  return cmat,losses

def main(_):
  run_best = 0
  best = 0

  # Remove tensorlogs
  try:
    if not FLAGS.restore:
      shutil.rmtree(FLAGS.run_dir + 'tensorlogs/')
  except FileNotFoundError:
    pass

  tb = Process(target=launchTensorBoard)
  tb.start()
  while True:
    max_lap  = 20
    overlap  = 0

    best_met = 0
    this_met = .01

    best = 0
    x = 1 if FLAGS.restore else 0


    while(this_met < best_met or overlap < max_lap):

      train(restore = (x != 0),epoch = x)
      cmat,loss = train(train_run = False, restore = False,epoch = x)
      this_met = loss

      if x == 0:
        best_met = this_met

      if best_met < this_met:
        overlap += 1
      else:
        best      = x
        overlap   = 0
        best_met = this_met
      x += 1
      print("\rEpoch %d, Best %d, Overlap %d"%(x,best,overlap))
      print("Best Metric: %.2f This metric: %.2f"%(best_met,this_met))

    if best > run_best:
      try:
        shutil.rmtree(FLAGS.run_dir + 'bestlogs/')
      except FileNotFoundError:
        pass
      shutil.copytree(FLAGS.run_dir + 'tensorlogs/',FLAGS.run_dir + 'bestlogs/')
  tb.terminate()
  tb.join()


if __name__ == '__main__':
  tf.app.run()
