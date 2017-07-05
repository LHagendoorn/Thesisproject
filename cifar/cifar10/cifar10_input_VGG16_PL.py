# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 224

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
classCount = 2
clustCount = 6

#PERC = 0.1

#t = np.zeros((11,2,6))
#t[1,0,0] = 1
#t[2,0,1] = 1
#t[3,1,0] = 1
#t[4,1,1] = 1
#t[5,1,2] = 1
#t[6,1,3] = 1
#t[7,1,4] = 1
#t[8,1,5] = 1
#t[9,0,2] = 1
#t[10,0,3] = 1

#lbl_lookup = {
#  tf.constant(0): tf.constant(t[0]),
#  tf.constant(1): tf.constant(t[1]),
#  tf.constant(2): tf.constant(t[2]),
#  tf.constant(3): tf.constant(t[3]),
#  tf.constant(4): tf.constant(t[4]),
#  tf.constant(5): tf.constant(t[5]),
#  tf.constant(6): tf.constant(t[6]),
#  tf.constant(7): tf.constant(t[7]),
#  tf.constant(8): tf.constant(t[8]),
#  tf.constant(9): tf.constant(t[9]),
#  tf.constant(10): tf.constant(t[10])
#}

VGG_MEAN = [103.939, 116.779, 123.68]

def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  labelled_bytes = 1 #byte to switch the labels of with partially labelled learning
  result.height = 32
  result.width = 32
  result.depth = 3
  #['airplane','automobile','bird','cat','deer','dog','frog','horse', 'ship','truck']

# [ 0 1 2 3 4 5 ] superclass1
# [ 6 7 8 9 10 11] superclass2

  supLab =      tf.constant([0,0,1,1,1,1,1,1,0,0]) #Man mande vs animals
  subclassLab = tf.constant([0,1,6,7,8,9,10,11,2,3])
  #supLab = tf.constant([0,0,1,1,1,1,1,0,0,0]) #Man mande+horse vs animals-horse
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes + labelled_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)
  #result.key = tf.decode_raw(result.key, tf.int32)
    
  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [labelled_bytes], [labelled_bytes+label_bytes]), tf.int32)
  print(result.label)
  result.switch = tf.cast(
      tf.strided_slice(record_bytes, [0], [labelled_bytes]), tf.int32)  
    
  result.superLabel = tf.cast(tf.gather(supLab,result.label), tf.int32)
  result.subclasslab = tf.cast(tf.gather(subclassLab,result.label), tf.int32)
  #result.superLabel = result.label
    
  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes+labelled_bytes],
                       [label_bytes + labelled_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label,superLabel, min_queue_examples,
                                    batch_size, shuffle, raw = False, raw_image = None):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
      images, label_batch, superLabel_batch = tf.train.shuffle_batch(
        [image, label, superLabel],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
      if not raw:
        images, label_batch, superLabel_batch = tf.train.batch(
            [image, label, superLabel],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
      if raw:
        images, raw_images, label_batch, superLabel_batch = tf.train.batch(
            [image, raw_image, label, superLabel],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
        return images, raw_images, label_batch, tf.reshape(superLabel_batch, [batch_size])

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  
  print(label_batch)
  return images, label_batch, tf.reshape(superLabel_batch, [batch_size])


def distorted_inputs(data_dir, batch_size,partially_labelled=False,matrix_lab=True):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_%d_10perc.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  size = tf.constant([IMAGE_SIZE,IMAGE_SIZE])
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(reshaped_image)
    
  # Randomly crop a [height, width] section of the image.
  #distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
  distorted_image = tf.image.resize_images(reshaped_image, size)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  #distorted_image = tf.image.random_brightness(distorted_image,
  #                                             max_delta=63)
  #distorted_image = tf.image.random_contrast(distorted_image,
  #                                           lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_standardization(distorted_image) #NOTE!!!

  # Convert RGB to BGR
  red, green, blue = tf.split(axis=2, num_or_size_splits=3, value=distorted_image)
  assert red.get_shape().as_list()[:] == [IMAGE_SIZE, IMAGE_SIZE, 1]
  assert green.get_shape().as_list()[:] == [IMAGE_SIZE, IMAGE_SIZE, 1]
  assert blue.get_shape().as_list()[:] == [IMAGE_SIZE, IMAGE_SIZE, 1]
  float_image = tf.concat(axis=2, values=[
      blue - VGG_MEAN[0],
      green - VGG_MEAN[1],
      red - VGG_MEAN[2],
  ])

  #clear labels
  if partially_labelled:
    if matrix_lab:
  #    print(read_input.key)
      #read_input.label = lbl_lookup[tf.add(read_input.label,tf.constant(1))*read_input.switch]
      #Makes lbl positive when switch=1 and negative when switch=0, negative values are all zeros after one_hot
      subclasslab_enc = tf.one_hot(read_input.subclasslab * (tf.constant(-1)+read_input.switch*tf.constant(2)), classCount*clustCount)
      read_input.label = tf.reshape(subclasslab_enc,(classCount,clustCount))
    else:
      read_input.label = tf.one_hot(read_input.label * (tf.constant(-1)+read_input.switch*tf.constant(2)), NUM_CLASSES)


  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  #read_input.label.set_shape([1])
  read_input.superLabel.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
  
  img, lbl, suplbl = _generate_image_and_label_batch(float_image, read_input.label, read_input.superLabel,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
  # Generate a batch of images and labels by building up a queue of examples.
  return img, lbl, suplbl


def inputs(eval_data, data_dir, batch_size,partially_labelled=False):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  size = tf.constant([IMAGE_SIZE,IMAGE_SIZE])
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_images(reshaped_image, size)

  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_standardization(resized_image) #NOTE!!!
  #rgb_scaled = resized_image * 255.0
  rgb_scaled = resized_image

  # Convert RGB to BGR
  red, green, blue = tf.split(axis=2, num_or_size_splits=3, value=rgb_scaled)
  assert red.get_shape().as_list()[:] == [IMAGE_SIZE, IMAGE_SIZE, 1]
  assert green.get_shape().as_list()[:] == [IMAGE_SIZE, IMAGE_SIZE, 1]
  assert blue.get_shape().as_list()[:] == [IMAGE_SIZE, IMAGE_SIZE, 1]
  bgr = tf.concat(axis=2, values=[
      blue - VGG_MEAN[0],
      green - VGG_MEAN[1],
      red - VGG_MEAN[2],
  ])
    
  #clear labels
  if partially_labelled:
      print(read_input.key)
      read_input.label = lbl_lookup[tf.add(read_input.label,tf.constant(1.0))*tf.gather_nd(KEEPLAB,read_input.key)]
          
    
  # Set the shapes of tensors.
  bgr.set_shape([height, width, 3])
  read_input.label.set_shape([1])
  read_input.superLabel.set_shape([1])

  print(read_input.label)
    
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(bgr, read_input.label, read_input.superLabel,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

def inputs_raw(eval_data, data_dir, batch_size,partially_labelled=False):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  size = tf.constant([IMAGE_SIZE,IMAGE_SIZE])
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_images(reshaped_image, size)

  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_standardization(resized_image) #NOTE!!!
  #rgb_scaled = resized_image * 255.0
  rgb_scaled = resized_image
    
  # Convert RGB to BGR
  red, green, blue = tf.split(axis=2, num_or_size_splits=3, value=rgb_scaled)
  assert red.get_shape().as_list()[:] == [224, 224, 1]
  assert green.get_shape().as_list()[:] == [224, 224, 1]
  assert blue.get_shape().as_list()[:] == [224, 224, 1]
  bgr = tf.concat(axis=2, values=[
      blue - VGG_MEAN[0],
      green - VGG_MEAN[1],
      red - VGG_MEAN[2],
  ])
  
  #clear labels
  if partially_labelled:
      print(read_input.key)
      read_input.label = lbl_lookup[tf.add(read_input.label,tf.constant(1.0))*tf.gather_nd(KEEPLAB,read_input.key)]
        

  # Set the shapes of tensors.
  bgr.set_shape([height, width, 3])
  resized_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])
  read_input.superLabel.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  img,img_raw,lbls, suplbls = _generate_image_and_label_batch(bgr, read_input.label, read_input.superLabel,
                                         min_queue_examples, batch_size,
                                         shuffle=False, raw=True, raw_image=reshaped_image)
  return img, img_raw, lbls, suplbls