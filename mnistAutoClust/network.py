# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:31:00 2017

@author: hagendoornl1
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets('C:/Users/hagendoornl1/Documents/TensorflowTuturial/MNIST_data', one_hot=True)

#%% Model

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x,W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def special_pool(x):
  return tf.divide(tf.reduce_max(tf.exp(x),1), tf.stack([tf.reduce_sum(tf.exp(x),[1,2]),tf.reduce_sum(tf.exp(x),[1,2])],1))

def zBar(x):
  return tf.maximum(tf.concat(x,1),0)
    
def bigU(x):
  return tf.matmul(tf.transpose(x),x)

def coactivity(x):
  #Select everything not in the diagonal:
  selection = np.ones(x.shape,dtype='float32') - np.eye(x.shape[1],dtype='float32')
  return tf.reduce_sum(tf.matmul(x,selection))

def bigV(x):
  smallNu=tf.reshape(tf.reduce_sum(x,axis=0),(1,-1))
  return tf.matmul(tf.transpose(smallNu),smallNu)

def specialNormalise(x):
  selection = np.ones(x.shape,dtype='float32') - np.eye(x.shape[0],dtype='float32')
  top = tf.reduce_sum(tf.matmul(x,selection))
  bottom = tf.multiply(tf.to_float(x.shape[0]-1),tf.reduce_sum(tf.matmul(x,np.eye(x.shape[0],dtype='float32'))))
  return tf.divide(top,bottom)

x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1,28,28,1])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

keep_prob = tf.placeholder(tf.float32)

W_softmax1 = weight_variable([1024, 5])
b_softmax1 = bias_variable([5])

W_softmax2 = weight_variable([1024, 5])
b_softmax2 = bias_variable([5])

W_softmaxgroup = weight_variable([2,2])
b_softmaxgroup = bias_variable([2])

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Two FC softmax layers
sm1 = tf.nn.softmax(tf.matmul(h_fc1, W_softmax1) + b_softmax1)
sm2 = tf.nn.softmax(tf.matmul(h_fc1, W_softmax2) + b_softmax2)

bZ = zBar([sm1,sm2])
bU = bigU(bZ)
coact = coactivity(bU)
affinity = specialNormalise(bU)

bV=bigV(bZ)
balance = specialNormalise(bV)

sm1_pool = tf.reduce_mean(sm1,1)
sm2_pool = tf.reduce_mean(sm2,1)

#pool to final classification
softmaxStacked = tf.stack([sm1_pool, sm2_pool],1)
y_conv=tf.nn.softmax(tf.matmul(softmaxStacked, W_softmaxgroup) + b_softmaxgroup)
#y_conv = special_pool(softmaxStacked)
#y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


cross_entropy = tf.reduce_max(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

one = tf.constant(0.2)
zero = tf.constant(0.0)

tresh = tf.constant(0.01)
c1 = one
c2 = one
c3 = tf.cond(tf.less(affinity,tresh),lambda: one,lambda: zero)
c4 = one

loss = cross_entropy # + c1*affinity + c2*(1-balance) + c3*coact + c4*tf.square(tf.norm(tf.concat([sm1,sm2],1),ord='fro',axis=(0,1)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

y = {0:[0,1], 1:[1,0]}

totalSteps = 2000
batchSize = 100
for i in range(totalSteps):
  trainbatch = mnist.train.next_batch(batchSize)
  trainbatch = (trainbatch[0],np.array([y[np.argmax(trainbatch[1][i])>4] for i in range(len(trainbatch[1]))]))
  valbatch = mnist.validation.next_batch(batchSize)
  valbatch = (valbatch[0],np.array([y[np.argmax(valbatch[1][i])>4] for i in range(len(valbatch[1]))]))
  if i%100 == 0:
    train_accuracy = loss.eval(feed_dict={
        x:trainbatch[0], y_: trainbatch[1], keep_prob: 1.0})
    val_accuracy = loss.eval(feed_dict={x:valbatch[0], y_:valbatch[1], keep_prob: 1.0})
    print("step %d/%d, training accuracy: %g, validation accuracy: %g"%(i,totalSteps, train_accuracy, val_accuracy))
    
    af = affinity.eval(feed_dict={x:trainbatch[0], y_: trainbatch[1], keep_prob: 1.0})
    ba = balance.eval(feed_dict={x:trainbatch[0], y_: trainbatch[1], keep_prob: 1.0})
    co = coact.eval(feed_dict={x:trainbatch[0], y_: trainbatch[1], keep_prob: 1.0})
    print("affinity: %g, balance: %g, coact: %g"%(af,(1-ba),co))
    #print("step %d, training loss %g"%(i, loss))
    feed_dict = {x: trainbatch[0], y_: trainbatch[1], keep_prob: 0.5}
  _, res_softMaxStacked, res_sm1, res_sm2 = sess.run([train_step,softmaxStacked,sm1,sm2],feed_dict=feed_dict)



#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))