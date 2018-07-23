from PIL    import Image
from string import ascii_letters, digits
import numpy as np, tensorflow as tf
import os, io, random, cv2, sys, pickle, contextlib
from multiprocessing import Process, Queue, Event

sys.path.append('E:/Wavelet-Upsampling/Helpers/google_images_download/')
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

    self.in_queue = Queue()

    # Find all images
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

    # Start the downloader. It'll self-terminate when it doesn't need to download any more.
    self.downloader = DownloaderProcess(self.in_queue, DATASET_TARGET_SIZE - len(self.internal_list))
    self.downloader.start()

    # Build the number of current data, the total target size, and the number seen
    #   in current epoch.
    self.num_examples = len(self.internal_list)
    self.tot_examples = DATASET_TARGET_SIZE
    self.num_seen     = 0

  # Write the three data lists.
  def write_lists(self,file_list):
    data_len = len(file_list)
    random.shuffle(file_list)
    self.file_writer('test' ,file_list[0                  :data_len * 3  // 10])
    self.file_writer('train',file_list[data_len * 3 // 10 :data_len * 9  // 10])
    self.file_writer('val'  ,file_list[data_len * 9 // 10 :data_len * 10 // 10])

  def file_writer(self,split,list):
    with open(self.base_directory + split + '.lst','wb') as fp:
      pickle.dump(list,fp)

  def file_reader(self,directory,split):
    with open(self.base_directory + split + '.lst','rb') as fp:
      list = pickle.load(fp)
    return list

  def get_next_batch(self,batch_size):
    # If batch_zie + num_seen > num_examples, just return
    #   examples to end of list.

    if batch_size+self.num_seen > self.tot_examples:
      print("OUT OF RANGE ON %d EXAMPLES"%len(self.internal_list))
      self.num_seen = 0
      random.shuffle(self.internal_list)
      raise IndexError

    # [print(self.img_directory+file+'.jpg') for file in batch_list]

    imgs = []
    files= []
    x = 0
    while len(imgs) < batch_size:

      # Pull all items out out of the queue until there are enough for a batch.
      while self.in_queue.qsize() > 0 or batch_size + self.num_seen > self.num_examples:
        self.internal_list.append(self.in_queue.get())
        self.num_examples = len(self.internal_list)

      # print(x,end=',')
      file = self.internal_list[self.num_seen+x]
      try:
        img = cv2.imread(self.img_directory+file+'.jpg')
        height, width, channels = img.shape
        if height < FLAGS.imgH or width < FLAGS.imgW:
          raise IndexError
        img = cv2.resize(img,(FLAGS.imgW,FLAGS.imgH))
        # print(np.unique(img))
        # self.test([img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        files.append(file)
      except:
        del self.internal_list[self.num_seen + x]
        self.num_examples = len(self.internal_list)
        x -= 1
      x += 1

    self.num_seen += batch_size
    return imgs,files

  def test(self,imgs):
    for img in imgs:
      cv2.imshow("Google",img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

class DownloaderProcess(Process):
  def __init__(self,q,to_download,):
    Process.__init__(self)
    self.exit        = Event()
    self.out_queue   = q
    self.to_download = to_download
    self.downloaded  = 0

  def run(self):
    while not self.exit.is_set():
      download_runner(self.out_queue)
      self.downloaded += 1
      if self.downloaded > self.to_download:
        self.shutdown()

  def shutdown(self):
    self.exit.set()

def download_runner(q):
  randomstr = get_rand_str()
  with contextlib.redirect_stdout(io.StringIO()):
    dls = downloader.download({'keywords':randomstr,'limit':RELATED_IMGS,'format':'jpg','color_type':'full-color','size':'>800*600','aspect_ratio':'wide','type':'photo','output_directory':'E:/Wavelet-Upsampling/Data/','image_directory':'./','safe_search':'','no_numbering':''})
  for key in dls:
    for path in dls[key]:
      q.put(path.split('\\')[-1].replace('.jpg',""))

if __name__ == '__main__':
  internal_generator = DataGenerator('train','E:/Wavelet-Upsampling/')
  imgs, batch_list = internal_generator.get_next_batch(20)
  internal_generator.test(imgs)
  internal_generator.downloader.terminate()
  internal_generator.downloader.join()
