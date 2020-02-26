from accumulator import Accumulator
import numpy as np
import tensorflow as tf

def str2list(s):
  return s.replace(" ","").split(',')

# for gradient clipping
def get_train_op(optim, loss, global_step=None, clip=None, var_list=None):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  grad_and_vars = optim.compute_gradients(loss, var_list=var_list)
  if clip is not None:
    grad_and_vars = [((None if grad is None \
        else tf.clip_by_value(grad, clip[0], clip[1])), var) \
        for grad, var in grad_and_vars]
  train_op = optim.apply_gradients(grad_and_vars, global_step=global_step)
  return train_op

def print_balancing_variables(
    args, flag, episode, ntask, omega, gamma, logfile):

  line = '\n%s '%(flag)
  print(line)
  logfile.write(line + '\n')

  # index sorting according to task size
  y = episode[1]
  n = []
  for t in range(ntask):
    y_t = y[t]
    y_t = y_t[np.sum(y_t,1)==1]
    n.append(y_t.shape[0])
  n = np.stack(n)
  idx = np.argsort(n)

  if args.gamma_on:
    line = '\n*** Gamma for task imbalance ***'
    print(line)
    logfile.write(line + '\n')

    line = '             '
    for i in range(1,5):
      line += ' conv%d' % i
    line += ' dense'
    print(line)
    logfile.write(line + '\n')

    line = ''
    for t in list(idx):
      line = 'task %d: N=%3d ' % (t, n[t])
      for i in range(5):
        line += '%.3f ' % gamma[t][i]
      print(line)
      logfile.write(line + '\n')

  if args.omega_on:
    line = '\n*** Omega for class imbalance ***'
    print(line)
    logfile.write(line + '\n')

    line = '       '
    for i in range(1,6):
      line += '  C%d' % i
    line += '    '
    for i in range(1,6):
      line += '    C%d' % i
    print(line)
    logfile.write(line + '\n')

    y = episode[1]
    for t in list(idx):
      # index sorting according to class size
      y_t = y[t]
      y_t = y_t[np.sum(y_t,1)==1]
      y_t = np.argmax(y_t, -1)
      n_c = np.stack([np.sum(y_t==c) for c in range(args.way)])
      idx_c = np.argsort(n_c)
      o = omega[t][idx_c]
      line = 'task %d: ' % (t)
      for c in range(args.way):
        line += '%3d ' % n_c[idx_c][c]
      line += '--> '
      for c in range(args.way):
        line += '%.3f ' % o[c]
      print(line)
      logfile.write(line + '\n')
