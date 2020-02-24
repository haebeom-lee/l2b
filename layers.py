import tensorflow as tf
import numpy as np

# functions
log = lambda x: tf.log(x + 1e-20)
softmax = tf.nn.softmax
relu = tf.nn.relu
softplus = tf.nn.softplus
sigmoid = tf.sigmoid
exp = tf.exp

# distributions
normal = tf.distributions.Normal

def kl_diagnormal_stdnormal(q):
  qshape = q.mean().shape
  p = normal(tf.zeros(qshape), tf.ones(qshape))
  return tf.distributions.kl_divergence(q, p)

# layers
batch_norm = tf.contrib.layers.batch_norm
flatten = tf.layers.flatten

def dense(x, dim, **kwargs):
  return tf.layers.dense(x, dim,
      kernel_initializer=tf.random_normal_initializer(stddev=0.02),
      bias_initializer=tf.zeros_initializer(), **kwargs)

def conv(x, filters, kernel_size=3, strides=1, **kwargs):
  return tf.layers.conv2d(x, filters, kernel_size, strides, padding='same',
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      bias_initializer=tf.zeros_initializer(), **kwargs)

def pool(x, **kwargs):
  return tf.layers.max_pooling2d(x, 2, 2, padding='valid', **kwargs)

# blocks
def conv_block(x, w, b, bn_scope='conv_bn'):
  x = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME') + b # NHWC
  x = batch_norm(x, activation_fn=relu, scope=bn_scope, reuse=tf.AUTO_REUSE)
  x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')
  return x

def dense_block(x, w, b):
  x = tf.matmul(flatten(x), w) + b
  return x

# training modules
def cross_entropy(logits, labels):
  losses = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=logits, labels=labels)
  return tf.reduce_mean(losses)

def cross_entropy_perclass(logits, labels):
  losses = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=logits, labels=labels)
  perclass = tf.matmul(tf.transpose(labels), tf.expand_dims(losses,1))
  N_t = tf.cast(tf.shape(labels)[0], dtype=tf.float32)
  way = tf.cast(tf.shape(labels)[1], dtype=tf.float32)
  return tf.squeeze(perclass) * way / N_t

def accuracy(logits, labels, axis=-1):
  correct = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
  return tf.reduce_mean(tf.cast(correct, tf.float32), axis)
