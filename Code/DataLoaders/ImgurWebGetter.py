from PIL import Image
from string import ascii_letters, digits
import numpy as np, tensorflow as tf
import os, random, pickle, json, cv2, inspect, imgurpython
from urllib.request import urlretrieve

client_id     = '81922f8805a2041'
client_secret = '14e8367cb48c6d1dabab66ea1b676e48d380ad16'
client        = imgurpython.ImgurClient(client_id,client_secret)

selection = ascii_letters + digits

allowed_types = ['image/png' ,
                 'image/jpg' ,
                 'image/jpeg',
                 'image/bmp'
                ]

# Things to discard an image over: NSFW,             boolean True
#                                  bad aspect ratio, min(rat, 1/rat) < .5
#                                  bad size,         h*w < 200**2
#                                  animated,         boolean True

def get_rand_id():
  str = ""
  for x in range(5):
    str += random.choice(selection)
  return str

class DataGenerator:
  def __init__(self,split,base_directory):
    pass

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

      id = get_rand_id()
      img = client.get_image(id)
      while os.path.isfile(img.link.split('/')[-1]):
        img = client.get_image(id)


      anim_flag  = img.animated

      nsfw_flag  = img.nsfw

      view_flag  = img.views < 10

      # title_flag = img.title is None

      ratio      = img.height / img.width
      ratio      = min(ratio,1/ratio)
      ratio_flag = ratio < (.5)

      size       = img.height * img.width
      size_flag  = (size < (200**2)) or (size > (2000**2))
      # print("ANIM: %r NSFW: %r RATIO: %.2f SIZE: %d"%(anim_flag,nsfw_flag,ratio,size))
      if nsfw_flag or ratio_flag or size_flag or anim_flag or view_flag: #or title_flag:
        # print("FOUND UNSUITABLE IMAGE...")
        pass
      else:
        urlretrieve(img.link,img.link.split('/')[-1])
        dl_image = cv2.imread(img.link.split('/')[-1])
        x=0
        # break
    except (imgurpython.helpers.error.ImgurClientError, ZeroDivisionError):
      pass
  results = inspect.getmembers(img)
  [print(result) for result in results]
  cv2.imshow("Imgur",dl_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
