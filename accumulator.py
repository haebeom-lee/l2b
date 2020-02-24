from __future__ import print_function

class Accumulator():
  def __init__(self, *args):
    self.args = args
    self.argdict = {}
    for i, arg in enumerate(args):
      self.argdict[arg] = i
    self.sums = [0]*len(args)
    self.cnt = 0

  def accum(self, val):
    val = [val] if type(val) is not list else val
    val = [v for v in val if v is not None]
    assert(len(val) == len(self.args))
    for i in range(len(val)):
      self.sums[i] += val[i]
    self.cnt += 1

  def clear(self):
    self.sums = [0]*len(self.args)
    self.cnt = 0

  def get(self, arg, avg=True):
    i = self.argdict.get(arg, -1)
    assert(i is not -1)
    return (self.sums[i]/self.cnt if avg else self.sums[i])

  def print_(self, header=None, episode=None, it=None, time=None,
    logfile=None, do_not_print=[], as_int=[], avg=True):

    line = '' if header is None else header + ': '
    if episode is not None:
      line += ('episode %d, ' % episode)
    if it is not None:
      line += ('iter %d, ' % it)
    if time is not None:
      line += ('(%.3f secs), ' % time)

    args = [arg for arg in self.args if arg not in do_not_print]

    arg = []
    for arg in args[:-1]:
      val = self.sums[self.argdict[arg]]
      if avg:
        val /= self.cnt
      if arg in as_int:
        line += ('%s %d, ' % (arg, int(val)))
      else:
        line += ('%s %.4f, ' % (arg, val))
    val = self.sums[self.argdict[args[-1]]]
    if avg:
      val /= self.cnt
    if arg in as_int:
      line += ('%s %d, ' % (arg, int(val)))
    else:
      line += ('%s %.4f' % (args[-1], val))
    print(line)

    if logfile is not None:
      logfile.write(line + '\n')
      logfile.flush()
