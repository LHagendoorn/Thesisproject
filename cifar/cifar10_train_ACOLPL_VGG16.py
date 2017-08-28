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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os

import tensorflow as tf

import cifar10_ACOLPL_VGG16_selectedFeatures as cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/code/logs/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""") #NOTE
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")
"""
Fileappend: string that is appended to all files
steps: number of train steps
continueTraining: when true, continue from current checkpoint in cifar10_train (Stage 2 ordered ACOL-PL)
Part1: boolean indicating whether this is the first stage of training or not, when True, only use labelled images
filename: Files to use (overwrites defaults)
"""
def train(fileappend = '.bin', steps=50000, continueTraining = False, part1 = False, filename = ''):
  """Train CIFAR-10 for a number of steps."""
  if not filename=='':
    FLAGS.train_dir = '/code/logs/' + filename
    
  with tf.Graph().as_default():        
    latest_step = tf.contrib.framework.get_or_create_global_step()
    print('--------------')
    print(latest_step)
    
    #if continueTraining:
    #    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)    
    #    latest_step = tf.Variable(int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]),dtype=tf.float32)
    #    print(latest_step)
    
    # Get images and labels for CIFAR-10.
    with tf.device('/cpu:0'):
        images, labels, superLabels = cifar10.distorted_inputs(part1,fileappend=fileappend)#NOTE

    # Build a Graph that computes the logits predictions from the
    # inference model.
    smStacked, stackedClusts,softmaxMat = cifar10.inference(images,contTraining = continueTraining)

    # Calculate loss.
    #loss, balance, affinity, coact = cifar10.loss(smStacked, stackedClusts, softmaxMat, labels, superLabels)
    loss, balance, affinity, coact = cifar10.loss(smStacked, stackedClusts, softmaxMat, labels, superLabels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op,lr, varsToStore = cifar10.train(loss, latest_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs([loss,balance,affinity,coact,lr])  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results[0]
          balance_value = run_values.results[1]
          affinity_value = run_values.results[2]
          coact_value = run_values.results[3]  
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f, balance = %.5f, affinity = %.5f, coact = %.5f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,1-balance_value,affinity_value,coact_value,
                               examples_per_sec, sec_per_batch))
          #losshist.append(loss)
    saver = tf.train.Saver(varsToStore)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      
      if continueTraining:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
          # Restores from checkpoint
          saver.restore(mon_sess, ckpt.model_checkpoint_path)
          #optimistic_restore(mon_sess, ckpt.model_checkpoint_path)
          #mon_sess.run(tf.global_variables_initializer())
          # Assuming model_checkpoint_path looks something like:
          #   /my-favorite-path/cifar10_train/model.ckpt-0,
          # extract global_step from it.
          #mon_sess.run(global_step.assign(latest_step))
        
      while not mon_sess.should_stop():
        mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  
  print(argv)
  if argv[1] is not None:
    fileappend = argv[1]
  else:
    fileappend = '.bin'
  if argv[2] is not None:  
    steps = int(argv[2])
  else:
    steps = 50000
  if len(argv) > 3:
    continueTraining = (argv[3]=='True')
    print(continueTraining)
  else:
    continueTraining = False
  if len(argv) > 4:
    stage1 = (argv[4]=='True')
  else:
    stage1 = False
  if len(argv)>5:
    skipfile = argv[5]
  else:
    skipfile = 'data_batch_1perc_skip.bin'
    print('NOTE: defaulted to 1 perc skip!')


  if not continueTraining:
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.Rename(FLAGS.train_dir,FLAGS.train_dir+str(datetime.now()))
    tf.gfile.MakeDirs(FLAGS.train_dir) 
  else:
    #print('NOT COPY!')
    #tf.gfile.Copy(FLAGS.train_dir,FLAGS.train_dir+'_stage1_'+str(datetime.now()))
    os.system('cp -r ' + FLAGS.train_dir + ' ' + str.replace(FLAGS.train_dir+'_stage1_'+str(datetime.now()),' ','_'))
  print('Training with: '+ fileappend)
  train(fileappend, steps, continueTraining, stage1)
  #with open(FLAGS.train_dir + '/losshist.txt','wb') as f:
    #for l in losshist:
    #    f.write(l)

if __name__ == '__main__':
  tf.app.run()
