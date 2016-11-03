'''
NeuralGPUResult records what happened during one run of the NeuralGPU

ResultsRecord keeps track of the results during one stage of training.
'''

from __future__ import print_function

import numpy as np

from . import data_utils

class NeuralGPUResult(object):
  """Recover of result of a single batch, which is always on one task."""
  grad_norm = None
  back_update = None
  loss = None
  output = None
  layers = None
  attention = None

  def __init__(self, vals, inp, target, taskid):
    self.feed_out = vals
    self.__dict__.update(vals)
    self.input = inp
    self.target = target
    self.taskid = taskid

  def accuracy(self, nprint=0):
    mask = self.target > 0
    errors = mask * (self.target != np.argmax(self.output, axis=-1))
    return np.sum(errors), np.sum(mask), np.sum(np.any(errors, axis=1))

  @property
  def length(self):
    return (self.input[0,:,0] > 0).sum()

  @property
  def batch_size(self):
    return len(self.input)

  def __repr__(self):
    err, tot, seq_err = self.accuracy()
    return '<NeuralGPUResult: length=%s loss=%s bs=%s err=%s seq_err=%s>' % \
      (self.length, self.loss, self.batch_size, err, seq_err)

  def attention_by_layer(self):
    return self.attention.mean(axis=-1).round(3)

  def to_string(self, i=None):
    if i is None:
      return '\n\n'.join(self.to_string(i) for i in range(self.batch_size))
    inp, outp, targ = map(data_utils.to_string, (self.input[i], self.output[i].argmax(axis=-1), self.target[i]))
    ans = '\n'.join([inp, '-'*len(outp), outp, targ])
    if hasattr(self, 'probs'):
      ans = '%s\n%s' % (ans, self.probs[:,i].round(3))
    return ans

  def plot_attention(self, figname):
    import pylab
    for i in range(self.attention.shape[2]):
      for j in range(self.attention.shape[1]):
        pylab.plot(self.attention[:,j,i], color='rbgkyo'[j], alpha=0.2, marker='o')
    pylab.savefig(figname)

def plot_many_examples(sess, model, max_length, generator, batch_size,
                       dirpat):
  examples = [(l, generator.get_batch(l, batch_size)) for l in range(3, max_length+1)
              if generator.is_valid_length(l)]
  for l, example in examples:
    print(l)
    result = model.step(sess, example, False)
    result.plot_attention(dirpat % l)

class ResultsRecord(object):
  """Result from many runs of training, on many tasks"""
  def __init__(self, batch_size):
    self.batch_size = batch_size
    self.record_for_task = {}

  def feed(self, results, step_time, below_curriculum):
    taskid = results.taskid[0]
    assert(not(np.any(results.taskid != taskid)))
    if taskid not in self.record_for_task:
      self.record_for_task[taskid] = ResultsRecordPerTask(self.batch_size)
    self.record_for_task[taskid].feed(results, step_time, below_curriculum)

  def __str__(self):
    def fmt_attr(name, fmt, label, scale=1):
      return label + ' '  + '/'.join(fmt % (getattr(v, name)*scale)
                                     for v in self.record_for_task.values())
    stat_list = [fmt_attr('avg_ppx', '%.8f', 'ppx'),
                fmt_attr('avg_grad_norm', '%.8f', 'grad-norm'),
                fmt_attr('avg_step_time', '%s', 'step-time'),
                fmt_attr('avg_err', '%.2f', 'errors', 100),
                fmt_attr('avg_seq_err', '%.2f', 'seq-errors', 100),
                ]
    if hasattr(next(iter(self.record_for_task.values())), 'binary_gap'):
      stat_list.append(fmt_attr('avg_binary_gap', '%.3f', 'binary-gap'))
    return ' '.join(stat_list)

class ResultsRecordPerTask(object):
  """Result of many batches on a single task"""
  def __init__(self, batch_size):
    self.batch_size = batch_size

    self.loss = 0.
    self.err = 0.
    self.seq_err = 0.
    self.acc = 0.
    self.grad_norm = 0.
    self.num_batches = 0
    self.num_below = 0
    self.step_time = 0.
    self.total = 0.

  def feed(self, results, step_time, below_curriculum):
    self.num_batches += 1
    self.num_below += below_curriculum

    self.step_time += step_time
    self.grad_norm += results.grad_norm
    for key in ['binary_gap']:
      if hasattr(results, key):
        if not hasattr(self, key):
          setattr(self, key, 0)
        setattr(self, key, getattr(self, key) + getattr(results, key))
    if below_curriculum:
      self.loss += results.loss
      err, tot, seq_err = results.accuracy()
      self.err += err
      self.seq_err += seq_err
      self.total += tot

  @property
  def safe_num_below(self):
    # If we happen to not have any samples within the curriculum, don't crash
    return self.num_below or 1.

  @property
  def avg_binary_gap(self):
    return self.binary_gap / self.num_batches

  @property
  def avg_step_time(self):
    return self.step_time / self.num_batches

  @property
  def avg_grad_norm(self):
    return self.grad_norm / self.num_batches

  @property
  def avg_loss(self):
    return self.loss / self.safe_num_below

  @property
  def avg_ppx(self):
    return data_utils.safe_exp(self.loss / self.safe_num_below)

  @property
  def avg_err(self):
    return self.err / (self.total or 1)

  @property
  def avg_seq_err(self):
    return self.seq_err / (self.safe_num_below * self.batch_size)
