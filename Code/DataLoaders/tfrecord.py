from PIL import Image
import numpy as np
import tensorflow as tf
import os.path
import random
import h5py

# Basic model parameters as external flags.
imgW = int(1920//4)
imgH = int(1080//4)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('imgW'      , imgW                   ,'Image Width')
flags.DEFINE_integer('imgH'      , imgH                   ,'Image Height')
flags.DEFINE_integer('num_classes'      , 5                   ,'# Classes')

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE =      'train_5class.tfrecords'
VALIDATION_FILE = 'test_5class.tfrecords'

# Helper functions for defining tf types
def _bytes_feature(value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_example(img_raw,label_raw):
	example = tf.train.Example(features=tf.train.Features(feature={
			'image_raw': _bytes_feature(img_raw),
			'label_raw': _bytes_feature(label_raw)}))
	return example

def write_image_label_pairs_to_tfrecord(filename_pairs, tfrecords_filename):
		"""Writes given image/label pairs to the tfrecords file.
		The function reads each image/label pair given filenames
		of image and respective label and writes it to the tfrecord
		file.
		Parameters
		----------
		filename_pairs : array of tuples (img_filepath, label_filepath)
				Array of tuples of image/label filenames
		tfrecords_filename : string
				Tfrecords filename to write the image/label pairs
		"""
		writer = tf.python_io.TFRecordWriter(tfrecords_filename)
		print(tfrecords_filename)
		i = 0
		for img_path, label_path in filename_pairs:

				img = np.array(Image.open(img_path))
				img = Image.fromarray(img)
				img = img.resize((imgW,imgH),Image.NEAREST)
				img = np.asarray(img)

				labelarr = []
				with h5py.File(label_path) as label:
					try:
						labelbak = label.get('f')
						labelarr = np.array(labelbak)
						labelarr = labelarr - 1
					except TypeError:
						try:
							labelbak = label.get('IND')
							labelarr = np.array(labelbak)
							labelarr = labelarr - 1
						except TypeError:
							labelbak = label.get('ind')
							labelarr = np.array(labelbak)
							labelarr = labelarr - 1
					labelarr = np.transpose(labelarr)

				label = labelarr

				label = Image.fromarray(label)
				label = label.resize((imgW,imgH),Image.NEAREST)
				label = np.asarray(label)

				label.flags.writeable = True
				label[np.where(label == 0)] = 4 # Car  -> Grnd -> 3
				label[np.where(label == 1)] = 1 # Tree -> Tree -> 0
				label[np.where(label == 2)] = 2 # Watr -> Watr -> 1
				label[np.where(label == 3)] = 3 # Bldn -> Bldn -> 2
				label[np.where(label == 4)] = 4 # Grnd -> Grnd -> 3
				label[np.where(label == 5)] = 4 # Boat -> Grnd -> 3
				label[np.where(label == 6)] = 5 # Road -> Road -> 4
        # classes = [Tree Water Building Ground Road]

				label = label - 1

				img       =   img.astype(np.uint8)
				label     = label.astype(np.uint8)
				print(img.shape)
				print(label.shape)

				# f, (img_p, lab_p) = plt.subplots(2)
				# img_p.imshow(img)
				# lab_p.imshow(label * 255)
				# plt.show()

				img_raw   =   img.tobytes()
				label_raw = label.tobytes()

				example = get_example(img_raw,label_raw)
				writer.write(example.SerializeToString())

				img_raw   =   np.fliplr(img).astype(np.uint8).tobytes()
				label_raw = np.fliplr(label).astype(np.uint8).tobytes()

				example = get_example(img_raw,label_raw)
				writer.write(example.SerializeToString())

				img_raw   =   np.flipud(img).astype(np.uint8).tobytes()
				label_raw = np.flipud(label).astype(np.uint8).tobytes()

				example = get_example(img_raw,label_raw)
				writer.write(example.SerializeToString())

				img_raw   =   np.fliplr(np.flipud(img)).astype(np.uint8).tobytes()
				label_raw = np.fliplr(np.flipud(label)).astype(np.uint8).tobytes()

				example = get_example(img_raw,label_raw)
				writer.write(example.SerializeToString())
				i  = i + 4
		print("Processed " + str(i) + " images...")
		print("Done!")

		writer.close()


def read_image_label_pairs_from_tfrecord(tfrecords_filename):
		"""Return image/label pairs from the tfrecords file.
		The function reads the tfrecords file and returns image
		and respective label matrices pairs.
		Parameters
		----------
		tfrecords_filename : string
				filename of .tfrecords file to read from
		Returns
		-------
		image_label_pairs : array of tuples (img, label)
				The image and label that were read from the file
		"""

		image_label_pairs = []

		record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

		for string_record in record_iterator:

				example = tf.train.Example()
				example.ParseFromString(string_record)

				img_string = (example.features.feature['image_raw']
																			.bytes_list
																			.value[0])

				label_string = (example.features.feature['label_raw']
																		.bytes_list
																		.value[0])

				img_1d = np.fromstring(img_string, dtype=np.uint8)
				img    = img_1d.reshape((height, width, -1))

				label_1d = np.fromstring(label_string, dtype=np.uint8)

				# labels don't have depth (3rd dimension)
				# TODO: check if it works for other datasets
				label = label_1d.reshape((height, width))

				image_label_pairs.append((img, label))

		return image_label_pairs


def read_decode(tfrecord_filenames_queue, num_classes):
		"""Return image/label tensors that are created by reading tfrecord file.
		The function accepts tfrecord filenames queue as an input which is usually
		can be created using tf.train.string_input_producer() where filename
		is specified with desired number of epochs. This function takes queue
		produced by aforemention tf.train.string_input_producer() and defines
		tensors converted from raw binary representations into
		reshaped image/label tensors.
		Parameters
		----------
		tfrecord_filenames_queue : tfrecord filename queue
				String queue object from tf.train.string_input_producer()
		Returns
		-------
		image, label : tuple of tf.int32 (image, label)
				Tuple of image/label tensors
		"""

		reader = tf.TFRecordReader()

		_, serialized_example = reader.read(tfrecord_filenames_queue)

		features = tf.parse_single_example(
			serialized_example,
			features={
				'image_raw': tf.FixedLenFeature([], tf.string),
				'label_raw': tf.FixedLenFeature([], tf.string)
				})


		image = tf.decode_raw(features['image_raw'], tf.uint8)
		label = tf.decode_raw(features['label_raw'], tf.uint8)

		intensity      = tf.convert_to_tensor(255,dtype=tf.float32,name='intensity')

		image_shape = tf.stack([imgH, imgW, 3])
		label_shape = tf.stack([imgH, imgW, 1])
		pre_hot_shp = tf.stack([imgH, imgW])
		hot_shape   = tf.stack([imgH, imgW, num_classes])

		# Normalize the image
		image = tf.cast(image,tf.float32)
		image = tf.divide(image,intensity)
		image = tf.subtract(image,.5)
		image = tf.reshape(image, image_shape)

		# Shape the label for the neural network
		label = tf.cast(label,tf.uint8)
		label = tf.reshape(label, pre_hot_shp)
		label_flat = label
		label = tf.one_hot(label,num_classes)
		label = tf.reshape(label, hot_shape)

		return image, label, label_flat

def inputs(train, batch_size, num_epochs, num_classes):
	filename = TRAIN_FILE if train else VALIDATION_FILE
	filename = "E:/BinaLab-Semantic-Segmentation/data/" + filename
	# print('\nInput file: %s'%filename)
	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

		image, label, label_flat = read_decode(filename_queue,num_classes)

		if train:
				images, labels, labels_flat = tf.train.shuffle_batch(
						[image, label, label_flat], batch_size=batch_size, num_threads=1,
						capacity=10 + 2 * batch_size,
						min_after_dequeue=10)
		else:
				images, labels, labels_flat = tf.train.batch(
						[image, label, label_flat], batch_size=batch_size, num_threads=2,
						capacity=10 + 2 * batch_size)
		return images, labels, labels_flat
