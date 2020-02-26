from layers import *
from encoder import InferenceNetwork

class LearningToBalance:
  def __init__(self, args):
    if args.id_dataset[0] in ['cifar', 'aircraft']:
      self.xdim, self.input_channel, self.n_channel = 32, 3, 32
    elif args.id_dataset[0] == 'mimgnet':
      self.xdim, self.input_channel, self.n_channel = 84, 3, 32
    else:
      raise ValueError("Invalid in-dist. dataset: %s" % args.id_dataset)

    self.numclass = args.way # num of classes per each episode
    self.n_steps = args.n_steps # num of inner gradient steps
    self.metabatch = args.metabatch
    self.inner_lr = args.inner_lr

    # placeholders. tr: train, te: test
    xshape = [None, None, (self.xdim**2)*self.input_channel]
    yshape = [None, None, self.numclass]
    self.episode = {
        'xtr': tf.placeholder(tf.float32, xshape, name='xtr'),
        'ytr': tf.placeholder(tf.float32, yshape, name='ytr'),
        'xte': tf.placeholder(tf.float32, xshape, name='xte'),
        'yte': tf.placeholder(tf.float32, yshape, name='yte')}

    # inference network to generate balancing variables
    self.encoder = InferenceNetwork(args)

    # turn on/off the learnable inner-stepsize vector (Meta-SGD)
    self.alpha_on = args.alpha_on

    # turn on/off the balancing variables
    self.omega_on = args.omega_on
    self.gamma_on = args.gamma_on
    self.z_on = args.z_on

  # get either 'theta' or 'alpha'
  # conventional 4-block conv net for few-shot learning
  def _get_param(self, param_name, reuse=None):

    # param initializers
    if param_name == 'theta':
      conv_init = tf.truncated_normal_initializer(stddev=0.02)
      fc_init = tf.random_normal_initializer(stddev=0.02)
      bias_init = tf.zeros_initializer()
    else:
      conv_init = fc_init = bias_init = \
          tf.constant_initializer(0.01)

    with tf.variable_scope(param_name, reuse=reuse):
      param = {}
      for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        param['conv%d_w'%l] = tf.get_variable('conv%d_w'%l,
            [3, 3, indim, self.n_channel], initializer=conv_init)
        param['conv%d_b'%l] = tf.get_variable('conv%d_b'%l,
            [self.n_channel], initializer=bias_init)

      remaining = 2*2 if self.xdim == 32 else 5*5
      if param_name == 'theta':
        param['dense_w'] = tf.zeros(
            [remaining*self.n_channel, self.numclass])
        param['dense_b'] = tf.zeros([self.numclass])
      else:
        single_w = tf.get_variable('dense_w',
            [remaining*self.n_channel, 1], initializer=fc_init)
        single_b = tf.get_variable('dense_b', [1], initializer=bias_init)
        param['dense_w'] = tf.tile(single_w, [1, self.numclass])
        param['dense_b'] = tf.tile(single_b, [self.numclass])
      return param

  # forward with model parameter theta
  def _forward(self, x, theta):
    x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])
    for l in [1,2,3,4]:
      w, b = theta['conv%d_w'%l], theta['conv%d_b'%l]
      x = conv_block(x, w, b, bn_scope='conv%d_bn'%l)
    w, b = theta['dense_w'], theta['dense_b']
    x = dense_block(x, w, b)
    return x

  # get placeholders
  def get_placeholders(self):
    return [self.episode['xtr'], self.episode['ytr'],
        self.episode['xte'], self.episode['yte']]

  # compute the outer objective w.r.t. a single task
  def forward_outer(self, inputs, sample, reuse=None):
    # theta: MAML initial model parameter
    # alpha: Meta-SGD learnable inner-stepsize vector
    theta = self._get_param('theta', reuse=reuse)
    if self.alpha_on:
      alpha = self._get_param('alpha', reuse=reuse)

    xtr, ytr, xte, yte = inputs

    # generate the three balancing variables
    omega, gamma, z, kl = self.encoder.forward(
        (xtr, ytr), sample, reuse=reuse)

    # scaling the KL term with training & test set size
    N_t = tf.cast(tf.shape(xtr)[0], dtype=tf.float32)
    M_t = tf.cast(tf.shape(xte)[0], dtype=tf.float32)
    kl = kl / (N_t + M_t)

    # [1] modulating MAML initialization with z
    if self.z_on:
      theta_update_z = {}
      for key in z.keys():
        if '_w' in key:
          theta_update_z[key] = theta[key] * (1. + z[key])
        elif '_b' in key:
          theta_update_z[key] = theta[key] + z[key]
        else:
          assert(False)
      theta.update(theta_update_z)

    # inner-gradient steps
    for i in range(self.n_steps):
      inner_logits = self._forward(xtr, theta)
      perclass = cross_entropy_perclass(inner_logits, ytr)

      # [2] modulating class-specific losses with omega
      # Note that this is equivalent to modulating class-specific gradients
      if self.omega_on:
        inner_loss = tf.reduce_sum(perclass * softmax(omega, -1))
      else:
        inner_loss = tf.reduce_mean(perclass)

      # compute inner-gradient
      grads = tf.gradients(inner_loss, list(theta.values()))
      gradients = dict(zip(theta.keys(), grads))

      # perform inner-gradient step
      theta_new = {}
      for key in theta.keys():
        if self.alpha_on:
          delta = alpha[key] * gradients[key]
        else:
          delta = self.inner_lr * gradients[key]

        # [3] modulating task-specific learning rates with gamma
        if self.gamma_on:
          theta_new[key] = theta[key] - delta * exp(gamma[key])
        else:
          theta_new[key] = theta[key] - delta
      theta.update(theta_new)

    # compute outer-loss and test accuracies
    logits_te = self._forward(xte, theta)
    cent = cross_entropy(logits_te, yte)
    acc = accuracy(logits_te, yte)
    pred = softmax(logits_te, -1)
    return cent, acc, kl, pred

  # compute the outer objective w.r.t. multiple tasks
  def forward_outer_multiple(self, sample, reuse=None):
    xtr, ytr = self.episode['xtr'], self.episode['ytr']
    xte, yte = self.episode['xte'], self.episode['yte']

    forward_outer_fn = lambda inputs: \
        self.forward_outer(inputs, sample=sample, reuse=reuse)

    cent, acc, kl, pred \
        = tf.map_fn(forward_outer_fn,
            elems=(xtr, ytr, xte, yte),
            dtype=(tf.float32, tf.float32, tf.float32, tf.float32),
            parallel_iterations=self.metabatch)

    net = {}
    net['weights'] = tf.trainable_variables()
    net['cent'] = tf.reduce_mean(cent)
    net['acc'] = acc
    net['kl'] = tf.reduce_mean(kl)
    net['pred'] = pred
    return net

  # MC integration for meta-testing
  def forward_outer_multiple_repeat(self, n_sample=10):
    preds = []
    for s in range(n_sample):
      pred = self.forward_outer_multiple(
          sample=True, reuse=True)['pred']
      preds.append(pred)
    preds = tf.reduce_mean(tf.stack(preds, 1), 1)
    return accuracy(preds, self.episode['yte'])

  # for printing tendency of the balancing variables
  def get_balancing_variables(self, ntask=10):
    x, y = self.episode['xtr'], self.episode['ytr']

    def ablation_fn(inputs):
      q_omega, q_gamma, q_z \
          = self.encoder.get_posterior(inputs, reuse=True)
      return q_omega.mean(), q_gamma.mean()

    omega, gamma \
        = tf.map_fn(ablation_fn,
            elems=(x, y),
            dtype=(tf.float32, tf.float32),
            parallel_iterations=self.metabatch)

    return softmax(omega, -1), exp(gamma)
