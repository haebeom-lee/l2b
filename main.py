from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import time
import os
from collections import OrderedDict

from model import LearningToBalance
from data import Data
from accumulator import Accumulator
from misc import get_train_op, print_balancing_variables, str2list

parser = argparse.ArgumentParser()

# meta-training / meta-testing
parser.add_argument('--mode', type=str,
    choices=['meta_train', 'meta_test'], default='meta_train')

# GPU id
parser.add_argument('--gpu_id', type=int, default=0)

# Save directory
parser.add_argument('--savedir', type=str, default=None)

# Save frequency
parser.add_argument('--save_freq', type=int, default=10000)

# Number of meta-training iterations
parser.add_argument('--n_train_iters', type=int, default=50000)

# Number of meta-testing episodes (or tasks)
parser.add_argument('--n_test_episodes', type=int, default=1000)

# In-distribution dataset
parser.add_argument('--id_dataset', type=str2list, default='cifar')

# Out-of-distribution dataset
parser.add_argument('--ood_dataset', type=str2list, default='svhn')

# Number of classes per each task
parser.add_argument('--way', type=int, default=5)

# Maximum number of shot (minimum shot is set to 1)
parser.add_argument('--max_shot', type=int, default=50)

# Number of test instances per class
parser.add_argument('--query', type=int, default=15)

# Number of MAML inner-gradient steps
parser.add_argument('--n_steps', type=int, default=5)

# Inner-gradient stepsize (when 'args.alpha' is turned off)
parser.add_argument('--inner_lr', type=float, default=0.5)

# Number of tasks per each meta-training step
parser.add_argument('--metabatch', type=int, default=4)

# outer learning rate
parser.add_argument('--meta_lr', type=float, default=1e-3)

# Whether to use Meta-SGD learnable inner-stepsize vector
parser.add_argument('--alpha_on', action='store_true', default=False)

# Whether to use each of the balancing variables (omega, gamma, z)
parser.add_argument('--omega_on', action='store_true', default=False)
parser.add_argument('--gamma_on', action='store_true', default=False)
parser.add_argument('--z_on', action='store_true', default=False)

# Number of MC samples for prediction (Only for meta-test)
parser.add_argument('--n_mc_samples', type=int, default=0)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

if not os.path.isdir(args.savedir):
  os.makedirs(args.savedir)


# for generating episode
id_data = OrderedDict([(data, Data(data)) for data in args.id_dataset])
ood_data = OrderedDict([(data, Data(data)) for data in args.ood_dataset])

# model object
model = LearningToBalance(args)
placeholders = model.get_placeholders()

# with sampling (for meta-training)
net = model.forward_outer_multiple(sample=True, reuse=False)
net_cent = net['cent']
net_kl = net['kl']
net_acc = net['acc']
net_acc_mean = tf.reduce_mean(net['acc'])

# without sampling (for meta-validation)
tnet = model.forward_outer_multiple(sample=False, reuse=True)
tnet_cent = tnet['cent']
tnet_kl = tnet['kl']
tnet_acc = tnet['acc']
tnet_acc_mean = tf.reduce_mean(tnet['acc'])
tnet_weights = tnet['weights']


def meta_train():
  global_step = tf.train.get_or_create_global_step()
  optim = tf.train.AdamOptimizer(args.meta_lr)
  outer_loss = net_cent + net_kl
  meta_train_op = get_train_op(optim, outer_loss, clip=[-3., 3.],
      global_step=global_step)

  saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
  logfile = open(os.path.join(args.savedir, 'train.log'), 'w')

  # flush out the arguments
  argdict = vars(args)
  print(argdict)
  for k, v in argdict.items():
    logfile.write(k + ': ' + str(v) + '\n')
  logfile.write('\n')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  meta_train_logger = Accumulator('cent', 'kl', 'id_acc')
  meta_train_to_run = [meta_train_op, net_cent, net_kl, net_acc_mean]

  # multi-dataset meta-validation loggers
  if len(args.id_dataset) > 1:
    meta_val_logger = OrderedDict([(data_name, Accumulator('cent', 'kl', 'id_acc'))
      for data_name in args.id_dataset + ['val']])
  # single dataset meta-validation logger
  else:
    meta_val_logger = { 'val': Accumulator('cent', 'kl', 'id_acc', 'ood_acc') }
  id_run = [tnet_cent, tnet_kl, tnet_acc_mean]
  ood_run = tnet_acc_mean

  for i in range(1, args.n_train_iters+1):
    # feed_dict
    data_name = args.id_dataset[i%len(args.id_dataset)]
    episode = id_data[data_name].generate_episode(
        args, split='mtr', n_episodes=args.metabatch)
    fd_mtr = dict(zip(placeholders, episode))
    meta_train_logger.accum(
        sess.run(meta_train_to_run, feed_dict=fd_mtr))

    if i % args.save_freq == 0:
      saver.save(sess, os.path.join(args.savedir, 'model-{}'.format(i)))

    if i % 100 == 0:
      line = 'Iter %d start, learning rate %f' % (i, args.meta_lr)
      print('\n' + line)
      logfile.write('\n' + line + '\n')
      meta_train_logger.print_(header='train', logfile=logfile)
      meta_train_logger.clear()

    # meta-validation
    if i % 10000 == 0:
      for j in range(3000 // args.metabatch):
        # valdate on in-distribution (ID) dataset(s)
        for data_name in args.id_dataset:
          id_episode = id_data[data_name].generate_episode(
              args, split='mval', n_episodes=args.metabatch)
          id_cent, id_kl, id_acc = sess.run(id_run,
              feed_dict=dict(zip(placeholders, id_episode)))

          if len(args.id_dataset) > 1:
            meta_val_logger[data_name].accum([id_cent, id_kl, id_acc])
            meta_val_logger['val'].accum([id_cent, id_kl, id_acc])

        # valiate on out-of-distribution (OOD) dataset
        if args.ood_dataset[0] in ['svhn', 'cub']:
          ood_episode = ood_data[args.ood_dataset[0]].generate_episode(
              args, split='mval', n_episodes=args.metabatch)
          ood_acc = sess.run(ood_run,
              feed_dict=dict(zip(placeholders, ood_episode)))
          meta_val_logger['val'].accum([id_cent, id_kl, id_acc, ood_acc])

      for data_name, logger in meta_val_logger.items():
          logger.print_(header='%s  '%data_name, logfile=logfile)
          logger.clear()

    # print balancing variables (omega, gamma) with 10 random tasks
    if args.gamma_on or args.omega_on:
      ntask = 10
      if i % 1000 == 0:
        testlist = [('ID ', id_data[args.id_dataset[0]])]
        if len(args.id_dataset) == 1:
          testlist.append(('OOD', ood_data[args.ood_dataset[0]]))
        for flag, data in testlist:
          # episode
          episode = data.generate_episode(
              args, split='mval', n_episodes=ntask)
          omega, gamma = sess.run(
              model.get_balancing_variables(ntask=ntask),
              feed_dict=dict(zip(placeholders, episode)))

          print_balancing_variables(
              args, flag, episode, ntask, omega, gamma, logfile)

  logfile.close()

def meta_test():
  # choose the best model
  f = open(os.path.join(args.savedir, 'train.log'), 'r')
  lines = [l.split(' ') for l in f.readlines()]
  acc_id, acc_ood, idx_list = [], [], []
  for i, l in enumerate(lines):
    if l[0] == 'val':
      if len(args.id_dataset) > 1:
        idx_list.append(int(lines[i-5][1]))
        acc_id.append(float(l[-1]))
      else:
        idx_list.append(int(lines[i-2][1]))
        acc_id.append(float(l[-3][:-1]))
        acc_ood.append(float(l[-1]))
  acc = np.array(acc_id) if len(args.id_dataset) > 1 \
          else np.array(acc_id) + np.array(acc_ood)
  best_idx = idx_list[np.argmax(acc)]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  saver = tf.train.Saver(tnet_weights, max_to_keep=10)
  saver.restore(sess, os.path.join(args.savedir, 'model-%d' % best_idx))

  logfile = open(os.path.join(args.savedir,
    'test_step%d_mc%d.log' % (args.n_steps, args.n_mc_samples)), 'w')
  start = time.time()

  line = '\nMeta-test model-%d\n' % best_idx
  print(line)
  logfile.write(line + '\n')

  if args.n_mc_samples > 0:
    to_run = model.forward_outer_multiple_repeat(n_sample=args.n_mc_samples)
  else:
    to_run = tnet_acc

  for flag, datalist in [('id ', id_data), ('ood', ood_data)]:
    for data_name, data in datalist.items():
      acc = []
      for j in range(args.n_test_episodes // args.metabatch):
        placeholders = model.get_placeholders()
        episode = data.generate_episode(
            args, split='mte', n_episodes=args.metabatch)
        acc.append(sess.run(
          to_run, feed_dict=dict(zip(placeholders, episode))))
        if (j+1)*args.metabatch % 100 == 0:
          print('(%.3f secs) test episode %d done' \
              % (time.time()-start, (j+1)*args.metabatch ))

      acc = 100.*np.concatenate(acc, axis=0)
      acc_mean = np.mean(acc)
      acc_95conf = 1.96*np.std(acc)/float(np.sqrt(args.n_test_episodes))
      result = '%s/%s acc: %.2f +- %.2f\n'%(flag, data_name, acc_mean, acc_95conf)
      print(result)
      logfile.write(result)

  logfile.close()

if __name__=='__main__':
  if args.mode == 'meta_train':
    meta_train()
  elif args.mode == 'meta_test':
    meta_test()
  else:
    raise ValueError('Invalid mode %s' % args.mode)
