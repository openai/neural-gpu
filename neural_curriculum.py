import data_utils
import numpy as np

class NeuralConfig(object):
  """Initial configuration settings for model"""

  config_keys = '''nmaps niclass noclass dropout rx_step max_grad_norm
  cutoff nconvs kw kh height mode lr pull pull_incr
  min_length batch_size grad_noise_scale task
  train_data_size init_weight curriculum_bound
  model_class input_height
  '''.split()

  def __init__(self, FLAGS, **kws):
    for key in self.config_keys:
      val = kws.get(key, getattr(FLAGS, key, None))
      setattr(self, key, val)

    min_length = 5
    max_length = min(FLAGS.max_length, data_utils.bins[-1])
    assert max_length + 1 > min_length
    self.max_length = max_length
    self.min_length = min_length
    self.input_height = self.height

  def __str__(self):
    msg1 = ("layers %d kw %d h %d kh %d relax %d batch %d noise %.2f task %s"
            % (self.nconvs, self.kw, self.height, self.kh, self.rx_step,
               self.batch_size, self.grad_noise_scale, self.task))
    msg2 = "data %d %s" % (self.train_data_size, msg1)
    msg3 = ("cut %.2f pull %.3f lr %.2f iw %.2f cr %.2f nm %d d%.4f gn %.2f %s" %
            (self.cutoff, self.pull_incr, self.lr, self.init_weight,
            self.curriculum_bound, self.nmaps, self.dropout, self.max_grad_norm, msg2))
    return msg3

class Curriculum(object):
  def __init__(self, generators, model_config):
    self.generators = generators
    for i, g in enumerate(generators):
      g.taskid = i

    self.min_length = model_config.min_length
    self.max_length = model_config.max_length
    self.model_config = model_config
    self.max_cur_lengths = {g.taskid: min(self.min_length+3, self.max_length)
                            for g in generators}

  def draw_length(self, cur_length, generator):
    l = None
    while l is None:
      # Select the length for curriculum learning.
      l = np.random.randint(self.min_length, cur_length + 1)
      if np.random.randint(100) < 60: # Prefer longer stuff 60% of time.
        l = max(l, np.random.randint(self.min_length, cur_length + 1))
      # Mixed curriculum learning: in 25% of cases go to an even larger length.
      if np.random.randint(100) < 25:
        l = max(l, np.random.randint(self.min_length, self.max_length + 1))

      if not generator.is_valid_length(l):
        l = None

    within_bounds = (l <= cur_length)
    return l, within_bounds

  def get_generator_for_task(self, task):
    return [g for g in self.generators if g.name == task][0]

  def test_examples(self, batch_size, task):
    generator = self.get_generator_for_task(task)
    for l in np.arange(self.min_length, self.max_length + 1):
      if generator.is_valid_length(l):
        yield (generator.get_batch(l, batch_size), l)

  def draw_example(self, batch_size, l=None, task=None):
    """Draw a random example"""
    generator = self.draw_generator(task)
    if l is None:
      cur_length = self.get_cur_length(generator)
      l, within_bounds = self.draw_length(cur_length, generator)
    else:
      within_bounds = True # XXX not clearly correct, but doesn't really matter
    return (generator.get_batch(l, batch_size), within_bounds)

  def tasks(self):
    """List of task names"""
    return [g.name for g in self.generators]

  def consider_extending(self, results):
    """Interpret the results"""
    pass

  def draw_generator(self, task=None):
    options = (self.generators if task is None else
               [g for g in self.generators if g.name == task])
    return np.random.choice(options)

  def get_cur_length(self, generator):
    return self.max_cur_lengths[generator.taskid]

  def consider_extending(self, record):
    ans = False
    for t in record.record_for_task:
      ans = max(ans, self.consider_extending_for_task(record.record_for_task[t], t))
    return ans

  def consider_extending_for_task(self, record, taskid):
    if record.avg_seq_err > self.model_config.curriculum_bound:
      return 0
    if self.max_cur_lengths[taskid] < self.max_length:
      self.max_cur_lengths[taskid] += 1
      while not self.generators[0].is_valid_length(self.max_cur_lengths[taskid]) and self.max_cur_lengths[taskid] < self.max_length:
        self.max_cur_lengths[taskid] += 1
      return 2
    return 1

  @property
  def length_str(self):
    return '/'.join(str(v) for k, v in sorted(self.max_cur_lengths.items()))

class GeneralizeCurriculum(Curriculum):

  def draw_generator(self, task=None):
    options = (self.generators[:1] if task is None else
               [g for g in self.generators if g.name == task])
    return options[0]

  @property
  def length_str(self):
    return str(self.max_cur_lengths[self.generators[0].taskid])

class BetterCurriculum(Curriculum):
  rand_prob = 0.2
  only_later = False
  decrease_threshold = 1

  def __init__(self, generators, model_config, kind):
    super(BetterCurriculum, self).__init__(generators, model_config)
    if kind == 2:
      self.decrease_threshold = 0.01
    elif kind == 3:
      self.rand_prob = 0
    elif kind == 4:
      self.only_later = True

  def draw_generator(self, task=None):
    if task is not None:
      return [g for g in self.generators if g.name == task][0]
    unsolved = [g for g in self.generators if self.max_cur_lengths[g.taskid] < self.max_length]
    if not unsolved:
      return np.random.choice(self.generators)
    if np.random.random() > self.rand_prob:
      return unsolved[0]
    if self.only_later:
      return np.random.choice(unsolved)
    else:
      return np.random.choice(self.generators)

  def consider_extending_for_task(self, record, taskid):
    if (self.max_cur_lengths[taskid] == self.max_length and
        record.avg_seq_err > self.decrease_threshold):
      self.max_cur_lengths[taskid] -= 1
      return 0
    if record.avg_seq_err > self.model_config.curriculum_bound:
      return 0
    val = super(BetterCurriculum, self).consider_extending_for_task(record, taskid)
    # Don't stop us from decreasing learning rate here
    if (self.max_cur_lengths[taskid] == self.max_length):
      return 0
    return val

class NeuralGPUResult(object):
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
    print l
    result = model.step(sess, example, False)
    result.attention = np.array(result.attention) #XXX kill soon
    result.plot_attention(dirpat % l)

class ResultsRecord(object):
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
    if hasattr(self.record_for_task.values()[0], 'binary_gap'):
      stat_list.append(fmt_attr('avg_binary_gap', '%.3f', 'binary-gap'))
    return ' '.join(stat_list)

class ResultsRecordPerTask(object):
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
