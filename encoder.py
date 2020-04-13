import tensorflow as tf
from layers import *

# inference network for generating the three balancing variables
class InferenceNetwork:
  def __init__(self, args):
    if args.id_dataset[0] == 'cifar' or len(args.id_dataset) > 1:
      self.xdim, self.input_channel, self.n_channel = 32, 3, 32
    elif args.id_dataset[0] == 'mimgnet':
      self.xdim, self.input_channel, self.n_channel = 84, 3, 32
    else:
      raise ValueError("Invalid in-dist. dataset: %s" % args.id_dataset)

    self.numclass = args.way
    self.max_shot = args.max_shot

    # turn on/off the balancing variables
    self.z_on = args.z_on
    self.gamma_on = args.gamma_on
    self.omega_on = args.omega_on

    self.s_on = True if len(args.id_dataset) > 1 else False
  # Compute element-wise sample mean, var., and set cardinality
  # then, return the concatenation of them.
  def _statistics_pooling(self, x, N):
    mean, var = tf.nn.moments(x, 0)
    N = tf.tile(tf.reshape(N, [-1]), mean.shape.as_list())
    return tf.stack([mean, var, N], 1)

  # compute the posterior of balancing variables
  def get_posterior(self, inputs, name='encoder', reuse=None):
    x, y = inputs

    # encoder 1
    x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])
    x = conv(x, 10, name=name+'/conv1', reuse=reuse)
    x = relu(x)
    x = pool(x)
    x = conv(x, 10, name=name+'/conv2', reuse=reuse)
    x = relu(x)
    x = pool(x)
    x = flatten(x)
    x = dense(x, 64, name=name+'/dense1', reuse=reuse)

    # statistics pooling 1
    ysum = tf.reduce_sum(y, 1)
    y_ = tf.argmax(y, 1)
    s = []; N_c_list = []
    for c in range(self.numclass):
      idx_c = tf.logical_and(tf.equal(y_,c), tf.greater(ysum, 0))
      x_c = tf.boolean_mask(x, idx_c) # x's corresponding to class c
      y_c = tf.boolean_mask(y, idx_c)
      N_c = (tf.reduce_sum(y_c)-1.)/(self.max_shot-1.) # normalized set size
      N_c_list.append(N_c)
      s_c = self._statistics_pooling(x_c, N=N_c)
      s.append(s_c)
    s = tf.stack(s, 0)
    s = dense(s, 4, name=name+'/interact1', reuse=reuse)
    s = relu(s)
    s = tf.reshape(s, [self.numclass, -1])

    # encoder 2
    v = dense(s, 128, name=name+'/dense2', reuse=reuse)
    v = relu(v)
    v = dense(v, 32, name=name+'/dense3', reuse=reuse)

    # statistics pooling 2
    v = self._statistics_pooling(v, N=tf.reduce_mean(N_c_list))
    v = tf.expand_dims(v, 0)
    v = dense(v, 4, name=name+'/interact2', reuse=reuse)
    v = relu(v)
    v = tf.reshape(v, [1, -1])

    # generate omega (from statistics pooling 1)
    s1 = dense(s, 64, name=name+'/dense_omega', reuse=reuse)
    s1 = relu(s1)
    odim = 1
    s_o = s if self.s_on else s1
    mu_omega = dense(s_o, odim, name=name+'/mu_omega', reuse=reuse)
    sigma_omega = dense(s_o, odim, name=name+'/sigma_omega', reuse=reuse)
    mu_omega, sigma_omega = tf.squeeze(mu_omega), tf.squeeze(sigma_omega)
    q_omega = normal(mu_omega, softplus(sigma_omega))


    # generate gamma (from statistics pooling 2)
    v1 = dense(v, 64, name=name+'/dense_gamma', reuse=reuse)
    v1 = relu(v1)
    gdim = 5
    mu_gamma = dense(v1, gdim, name=name+'/mu_gamma', reuse=reuse)
    sigma_gamma = dense(v1, gdim, name=name+'/sigma_gamma', reuse=reuse)
    mu_gamma, sigma_gamma = tf.squeeze(mu_gamma), tf.squeeze(sigma_gamma)
    q_gamma = normal(mu_gamma, softplus(sigma_gamma))

    # generate z (from statistics pooling 2)
    v2 = dense(v, 64, name=name+'/dense_z', reuse=reuse)
    v2 = relu(v2)
    zdim = 2*self.n_channel*4
    mu_z = dense(v2, zdim, name=name+'/mu_z', reuse=reuse)
    sigma_z = dense(v2, zdim, name=name+'/sigma_z', reuse=reuse)
    mu_z, sigma_z = tf.squeeze(mu_z), tf.squeeze(sigma_z)
    q_z = normal(mu_z, softplus(sigma_z))
    return q_omega, q_gamma, q_z

  def forward(self, inputs, sample, reuse=None):
    # compute posterior
    q_omega, q_gamma, q_z = self.get_posterior(inputs, reuse=reuse)

    # compute kl
    kl_omega = tf.reduce_sum(kl_diagnormal_stdnormal(q_omega))
    kl_gamma = tf.reduce_sum(kl_diagnormal_stdnormal(q_gamma))
    kl_z = tf.reduce_sum(kl_diagnormal_stdnormal(q_z))

    # sample variables from the posterior
    omega, gamma, z = None, None, None
    kl = 0.
    if self.omega_on:
      kl = kl + kl_omega
      omega = q_omega.sample() if sample else q_omega.mean()

    if self.gamma_on:
      kl = kl + kl_gamma
      g_ = q_gamma.sample() if sample else q_gamma.mean()
      g_ = tf.split(g_, [1,1,1,1,1], 0)
      gamma = {}
      for l in [1,2,3,4]:
        gamma['conv%d_w'%l] = gamma['conv%d_b'%l] = g_[l-1]
      gamma['dense_w'] = gamma['dense_b'] = g_[4]

    if self.z_on:
      kl = kl + kl_z
      z_ = q_z.sample() if sample else q_z.mean()
      zw_ = tf.split(z_[:self.n_channel*4], [self.n_channel]*4, 0)
      zb_ = tf.split(z_[self.n_channel*4:], [self.n_channel]*4, 0)
      z = {}
      for l in [1,2,3,4]:
        z['conv%d_w'%l] = zw_[l-1]
        z['conv%d_b'%l] = zb_[l-1]

    return omega, gamma, z, kl
