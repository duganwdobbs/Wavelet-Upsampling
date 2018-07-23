from PIL import Image
from string import ascii_letters, digits
import numpy as np, tensorflow as tf
import os, random, pickle, json, cv2, inspect
from urllib.request import urlretrieve

client_key    = 'fd41b315714112594d814d9c8c6c798a'
client_secret = 'c86ee71fcdfc3a00'

allowed_types = ['image/png' ,
                 'image/jpg' ,
                 'image/jpeg',
                 'image/bmp'
                ]

# var url = "https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key="
            # + flickrAPIKey
            # + "&license=" + currentLicenses.join()
            # + minDateString + maxDateString
            # + wordString
            # + useridString
            # + colors
            # + styles
            # + orientations
            # + safeSearch
            # + minSizeString
            # + mediaString
            # + "&per_page=" + itemsPerPage
            # + "&page=" + pageNumber
            # + "&format=json&nojsoncallback=1";


# Things to discard an image over: NSFW,             boolean True
#                                  bad aspect ratio, min(rat, 1/rat) < .5
#                                  bad size,         h*w < 200**2
#                                  animated,         boolean True

class DataGenerator:
  def __init__(self,split,base_directory):
    self.base_directory = base_directory
    self.data_directory = base_directory + 'data'
    # Build list of already found files.
    self.internal_list = [f.replace('.png','') for f in os.listdir(self.base_directory) if f.endswith('.png')]
    if (not os.path.isfile(self.data_directory + 'test.lst' ) or
        not os.path.isfile(self.data_directory + 'train.lst') or
        not os.path.isfile(self.data_directory + 'val.lst'  )   ):
        print("\rFILE SPLITS NOT FOUND... REBUILDING.",end='')
        data_len = len(self.internal_list)
        random.shuffle(self.internal_list)
        self.file_writer('test' ,file_list[0                  :data_len * 3  // 10])
        self.file_writer('train',file_list[data_len * 3 // 10 :data_len * 9  // 10])
        self.file_writer('val'  ,file_list[data_len * 9 // 10 :data_len * 10 // 10])

    test_list = file_reader(self,self.data_directory,'test.lst')
    train_list = file_reader(self,self.data_directory,'train.lst')
    val_list = file_reader(self,self.data_directory,'val.lst')

    total_len = len(test_list + train_list + val_list)
    if total_len < len(self.internal_list):
      print("\rFILE SPLITS NOT COMPLETE... REBUILDING.",end='')
      data_len = len(self.internal_list)
      random.shuffle(self.internal_list)
      self.file_writer('test' ,file_list[0                  :data_len * 3  // 10])
      self.file_writer('train',file_list[data_len * 3 // 10 :data_len * 9  // 10])
      self.file_writer('val'  ,file_list[data_len * 9 // 10 :data_len * 10 // 10])


    if split is 'train':
      self.examples = 6000
    if split is 'test':
      self.examples = 3000
    if testt is 'val':
      self.examples = 1000


  def file_writer(self,split,list):
    with open(self.base_directory + split + '.lst','wb') as fp:
      pickle.dump(list,fp)

  def file_reader(self,directory,split):
    with open(self.base_directory + split + '.lst','rb') as fp:
      list = pickle.load(fp)
    return list

  def get_next_batch(self,batch_size):
    pass

  def test(self):
    pass


if __name__ == '__main__':
  x = 0
  while True:
    try:
      x += 1
      print("\rATTEMPT %d     "%x,end='')

    except (ZeroDivisionError):
      pass
  results = inspect.getmembers(img)
  [print(result) for result in results]
  cv2.imshow("Imgur",dl_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
