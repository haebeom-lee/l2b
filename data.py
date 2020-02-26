import numpy as np
import os

class Data:
  def __init__(self, dataset_name):
    self.dataset_name = dataset_name
    path = os.path.join('data', dataset_name)

    if dataset_name in ['cifar', 'mimgnet', 'aircraft', 'quickdraw', 'vgg_flower']:
      # meta-training set
      x = np.load(os.path.join(path, 'train.npy'), encoding='bytes')
      self.C_mtr = len(x)
      self.N_mtr = [len(xx) for xx in x]
      self.x_mtr = [np.reshape(x[i], [self.N_mtr[i], -1]) for i in range(self.C_mtr)]

    # meta-validation set
    if not dataset_name in ['traffic', 'fashion-mnist']:
      x = np.load(os.path.join(path, 'valid.npy'), encoding='bytes')
      self.C_mval = len(x)
      self.N_mval = [len(xx) for xx in x]
      self.x_mval = [np.reshape(x[i], [self.N_mval[i], -1]) for i in range(self.C_mval)]

    # meta-test set
    x = np.load(os.path.join(path, 'test.npy'), encoding='bytes')
    self.C_mte = len(x)
    self.N_mte = [len(xx) for xx in x]
    self.x_mte = [np.reshape(x[i], [self.N_mte[i], -1]) for i in range(self.C_mte)]

  def generate_episode(self, args, split, n_episodes, mode=1):
    if split=='mtr':
      N, C, x = self.N_mtr, self.C_mtr, self.x_mtr
    elif split=='mval':
      N, C, x = self.N_mval, self.C_mval, self.x_mval
    elif split=='mte':
      N, C, x = self.N_mte, self.C_mte, self.x_mte
    else:
      raise ValueError('No such split: %s' % split)

    # metabatch
    xtr_list, ytr_list, xte_list, yte_list = [], [], [], []
    lenlist_tr, lenlist_te = [], []

    for t in range(n_episodes):

      # sample WAY number of classes
      classes = np.random.choice(
          range(C), size=args.way, replace=False)

      # both class and task imbalance (w/ half and half probability)
      if mode == 1:
        coin_toss = np.random.uniform(0, 1, 1)
        if coin_toss > 0.5:
          shot = np.random.choice(
              range(1, args.max_shot+1), size=args.way, replace=True)
        else:
          shot = np.random.choice(range(1, args.max_shot+1), size=1)
          shot = shot.repeat(args.way)

      # only class imbalance
      elif mode == 2:
        shot = np.random.choice(
            range(1, args.max_shot+1), size=args.way, replace=True)

      # only task imbalance
      elif mode == 3:
        shot = np.random.choice(range(1, args.max_shot+1), size=1)
        shot = shot.repeat(args.way)

      else:
        assert(False)

      xtr, ytr, xte, yte = [], [], [], []
      for i, c in enumerate(list(classes)):
        # sample SHOT + QUERY number of instances
        shot[i] = shot[i] if N[c] >= (shot[i]+args.query) \
                    else N[c] - args.query
        idx = np.random.choice(
            range(N[c]), size=shot[i]+args.query, replace=False)
        x_c = x[c][idx]
        xtr.append(x_c[:shot[i]])
        xte.append(x_c[shot[i]:])
        # make labels
        y = np.zeros(args.way); y[i] = 1.
        ytr.append(np.tile(np.expand_dims(y,0), [shot[i],1]))
        yte.append(np.tile(np.expand_dims(y,0), [args.query,1]))

      xtr = np.concatenate(xtr, 0)
      ytr = np.concatenate(ytr, 0)
      xte = np.concatenate(xte, 0)
      yte = np.concatenate(yte, 0)

      lenlist_tr.append(xtr.shape[0])

      xtr_list.append(xtr)
      ytr_list.append(ytr)
      xte_list.append(xte)
      yte_list.append(yte)

    # fill in zeros to match the xtr and ytr's shape across the episodes
    maxlen = max(lenlist_tr)
    for t in range(n_episodes):
      x = xtr_list[t]
      y = ytr_list[t]

      curlen = x.shape[0]
      x_pad = np.zeros([maxlen-curlen, x.shape[1]])
      y_pad = np.zeros([maxlen-curlen, y.shape[1]])

      xtr_list[t] = np.concatenate([x, x_pad], 0)
      ytr_list[t] = np.concatenate([y, y_pad], 0)

    xtr_all = np.stack(xtr_list, 0)
    ytr_all = np.stack(ytr_list, 0)
    xte_all = np.stack(xte_list, 0)
    yte_all = np.stack(yte_list, 0)

    return xtr_all, ytr_all, xte_all, yte_all

