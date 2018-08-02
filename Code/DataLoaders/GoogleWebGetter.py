from PIL    import Image
from string import ascii_letters, digits
import numpy as np, tensorflow as tf
import os, io, random, cv2, sys, pickle, contextlib
from multiprocessing import Process, Queue, Event

from google_images_download import google_images_download

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('imgW'       , 912                   ,'Image Width')
flags.DEFINE_integer('imgH'       , 600                   ,'Image Height')

downloader = google_images_download.googleimagesdownload()
selection = digits + ascii_letters
RELATED_IMGS = 5

DATASET_TARGET_SIZE = 10000

data_directory = 'E:/Wavelet-Upsampling/Data'

allowed_types = ['image/png' ,
                 'image/jpg' ,
                 'image/jpeg',
                 'image/bmp'
                ]

allowed_exts = ['.png',
                #'.jpg',
                #'.jpeg',
                '.bmp'
               ]

def get_rand_str():
  str = ""
  for x in range(5):
    str += random.choice(selection)
  #str += random.choice(allowed_exts)
  return str

class DataGenerator:
  def __init__(self,split,base_directory):
    self.base_directory = base_directory + '/'
    self.img_directory  = base_directory + '/Data/'
    self.epochs      = 0

    self.in_queue = Queue()

    # Find all images
    try:
      file_list = [f.replace('.jpg' ,'') for f in os.listdir(self.img_directory ) if f.endswith('.jpg' )]
    except FileNotFoundError:
      os.mkdir(self.img_directory)
      file_list = [f.replace('.jpg' ,'') for f in os.listdir(self.img_directory ) if f.endswith('.jpg' )]
    # If split lists don't exist as files, create them. Shuffle values, then
    # write to test,train,val.lst
    if (not os.path.isfile(base_directory + 'test.lst' ) or
        not os.path.isfile(base_directory + 'train.lst') or
        not os.path.isfile(base_directory + 'val.lst'  )   ):
      print("\rFILE SPLITS NOT FOUND... REBUILDING.",end='')
      self.write_lists(file_list)

    test_list  = self.file_reader(base_directory, 'test')
    train_list = self.file_reader(base_directory, 'train')
    val_list   = self.file_reader(base_directory, 'val')
    if len(file_list) != len(test_list) + len(train_list) + len(val_list):
      print("\rFILE SPLITS NOT ACCURATE... REBUILDING.",end='')
      self.write_lists(file_list)
    else:
      print('\rLISTS WERE FOUND TO BE ACCEPTABLE,  %d VS %d'%(len(file_list),len(test_list) + len(train_list) + len(val_list)))

    # Assign internal list
    self.internal_list = self.file_reader(base_directory, split)

    # Shuffle internal list order (Random Shuffle Batch)
    random.shuffle(self.internal_list)

    # Build the number of current data, the total target size, and the number seen
    #   in current epoch.
    self.num_examples = len(self.internal_list)
    self.tot_examples = max(DATASET_TARGET_SIZE,len(self.internal_list))
    self.num_seen     = 0

  # Write the three data lists.
  def write_lists(self,file_list):
    data_len = len(file_list)
    random.shuffle(file_list)
    self.file_writer('test' ,file_list[0                  :data_len * 3  // 10])
    self.file_writer('train',file_list[data_len * 3 // 10 :data_len * 9  // 10])
    self.file_writer('val'  ,file_list[data_len * 9 // 10 :data_len * 10 // 10])

  def start(self,num_threads = 5):
    # Start the downloader. It'll self-terminate when it doesn't need to download any more.
    self.downloaders = [DownloaderProcess(self.in_queue, DATASET_TARGET_SIZE - len(self.internal_list),self.img_directory) for x in range(num_threads)]
    [downloader.start() for downloader in self.downloaders]

  def stop(self):
    for downloader in self.downloaders:
      downloader.terminate()
      downloader.join()

  # A class helper function to write file lists.
  def file_writer(self,split,list):
    with open(self.base_directory + split + '.lst','wb') as fp:
      pickle.dump(list,fp)

  # A class helper function to read file lists.
  def file_reader(self,directory,split):
    with open(self.base_directory + split + '.lst','rb') as fp:
      list = pickle.load(fp)
    return list

  def get_next_batch(self,batch_size):
    # If batch_zie + num_seen > num_examples, just return
    #   examples to end of list.

    if batch_size+self.num_seen > self.tot_examples:
      self.epochs  += 1
      self.num_seen = 0
      random.shuffle(self.internal_list)
      # Test the number of epochs to see if we should stop
      if self.epochs >= FLAGS.num_epochs:
        self.stop()
        raise KeyError

    # [print(self.img_directory+file+'.jpg') for file in batch_list]

    imgs = []
    files= []
    x = 0
    while len(imgs) < batch_size:

      # Pull all items out out of the queue until there are enough for a batch.
      while self.in_queue.qsize() > 0 or batch_size + self.num_seen > self.num_examples:
        self.internal_list.append(self.in_queue.get())
        self.num_examples = len(self.internal_list)

      file = self.internal_list[self.num_seen+x]
      try:
        # Read the images
        img = cv2.imread(self.img_directory+file+'.jpg')
        height, width, channels = img.shape
        # If the image isn't big enough, raise an error to discard the image
        if height < FLAGS.imgH or width < FLAGS.imgW:
          raise IndexError

        # Resize the image to the network dimensions
        img = cv2.resize(img,(FLAGS.imgW,FLAGS.imgH))

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add the image to the list of current images
        imgs.append(img)

        # Add the file to the list of current files
        files.append(file)
      except:
        # If any errors occur, this block fires
        try:
          # Remove the bad file
          os.remove(self.img_directory+file+'.jpg')
        except:
          # If it can't remove the bad file, just ignore it for now.
          pass
        # Delete the bad file from the internal list
        del self.internal_list[self.num_seen + x]
        # Start a new file downloading.
        # newDLer = DownloaderProcess(self.in_queue, DATASET_TARGET_SIZE - len(self.internal_list),self.img_directory)
        # newDLer.start()
        # self.downloaders.append(newDLer)


        # Resize the number of examples in the current list.
        self.num_examples = len(self.internal_list)

        # Reduce the value of x so that we know how far we've gone.
        x -= 1
      x += 1

    self.num_seen += batch_size
    return imgs,files

  def test(self,imgs):
    for img in imgs:
      cv2.imshow("Google",img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

# This class runs the downloading threading, and inherits from Process
class DownloaderProcess(Process):
  def __init__(self,q,to_download,img_directory,):
    Process.__init__(self)
    self.img_directory = img_directory
    self.exit        = Event()
    self.out_queue   = q
    self.to_download = to_download
    self.downloaded  = 0

  # Overloaded inherited run functionality from multiprocessing.Process
  def run(self):
    while not self.exit.is_set():
      download_runner(self.out_queue,self.img_directory)
      self.downloaded += RELATED_IMGS
      if self.downloaded > self.to_download:
        self.shutdown()

  def shutdown(self):
    self.exit.set()

# This is the function to run the downloading.
def download_runner(q,img_directory):
  randomstr = get_rand_str()
  # Mute console output for this function
  with contextlib.redirect_stdout(io.StringIO()):
    dls = downloader.download({'keywords':randomstr,
                               'limit':RELATED_IMGS,
                               'format':'jpg',
                               'color_type':'full-color',
                               'size':'>800*600',
                               'aspect_ratio':'wide',
                               'type':'photo',
                               'output_directory':img_directory,
                               'image_directory':'./',
                               'safe_search':'',
                               'no_numbering':''                 })
  for key in dls:
    for path in dls[key]:
      q.put(path.split('\\')[-1].replace('.jpg',""))

# This is a debugging function that runs if this specific file is run.
if __name__ == '__main__':
  internal_generator = DataGenerator('train','D:/Wavelet-Upsampling/')
  imgs, batch_list = internal_generator.get_next_batch(20)
  internal_generator.test(imgs)
  internal_generator.downloader.terminate()
  internal_generator.downloader.join()
