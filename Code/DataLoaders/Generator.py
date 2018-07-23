from PIL import Image
import numpy as np, tensorflow as tf
import os, random, pickle, json

# Basic model parameters as external flags.
imgW = int(224)
imgH = int(224)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('imgW'       , imgW                   ,'Image Width')
flags.DEFINE_integer('imgH'       , imgH                   ,'Image Height')
flags.DEFINE_integer('num_classes', 5                      ,'Tentative class for object types')

class DataGenerator:
  def __init__(self,split,base_directory):
    self.base_directory = base_directory + '/'
    self.img_directory  = base_directory + '/images/'
    self.json_directory = base_directory + '/metadata/'

    # If split lists don't exist as files, create them. Shuffle values, then
    # write to test,train,val.lst
    if (not os.path.isfile(base_directory + 'test.lst' ) or
        not os.path.isfile(base_directory + 'train.lst') or
        not os.path.isfile(base_directory + 'val.lst'  )   ):
      print("\rFILE SPLITS NOT FOUND... REBUILDING.",end='')

      # Find all images
      file_list = [f.replace('.jpg' ,'') for f in os.listdir(self.img_directory ) if f.endswith('.jpg' )]
      json_list = [f.replace('.json','') for f in os.listdir(self.json_directory) if f.endswith('.json')]
      # Finding the intersection between two lists is MUCH FASTER than looking
      #   for every single file. This could theoretically error, if listdir
      #   pulls up more files than actually exist.

      file_list = list(set(file_list) & set(json_list))
      print("\r%d FILES FOUND                                "%(len(file_list)))

      data_len = len(file_list)
      random.shuffle(file_list)
      self.file_writer('test' ,file_list[0                  :data_len * 3  // 10])
      self.file_writer('train',file_list[data_len * 3 // 10 :data_len * 9  // 10])
      self.file_writer('val'  ,file_list[data_len * 9 // 10 :data_len * 10 // 10])

    # Assign internal list
    self.internal_list = self.file_reader(base_directory, split)

    # Shuffle internal list order (Random Shuffle Batch)
    random.shuffle(self.internal_list)

    # Necessary values: num_examples (len(list))
    self.num_examples = len(self.internal_list)
    #                   num_seen     (0)
    self.num_seen     = 0

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
    if batch_size+self.num_seen > self.num_examples:
      print("OUT OF RANGE ON %d EXAMPLES"%len(self.internal_list))
      self.num_seen = 0
      random.shuffle(self.internal_list)
      raise IndexError
    batch_list = self.internal_list[self.num_seen:self.num_seen+batch_size]
    imgs,anns  = self.load_imgs_annotations(batch_list)
    self.num_seen += batch_size
    return imgs,anns,batch_list

  def load_imgs_annotations(self,img_ids):
    imgs = []
    anns = []
    for img_id in img_ids:
      img = Image.open(self.img_directory  + img_id + '.jpg' )
      img = img.resize([imgW,imgH],Image.NEAREST)
      img = np.asarray(img)
      ann = self.load_json_data(self.json_directory + img_id + '.json')
      imgs.append(img)
      anns.append(ann)

    return imgs,anns

  def load_json_data(self,json_file):
    with open(json_file) as f:
      meta = json.load(f)
      return meta['EXPECTED_QUANTITY']

  def test(self):
    imgs,anns,batch_list = self.get_next_batch(5)
    print("Annotations: ",anns)
    print("IDs: ",batch_list)

if __name__ == '__main__':
  print("STARTING GENERATOR TEST")
  generator = DataGenerator('train','F:/amazon-data/')
  generator.test()
  generator.test()
