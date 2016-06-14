import data_utils
import numpy as np

class NeuralConfig(object):
  """Initial configuration settings for model"""

  config_keys = '''nmaps niclass noclass dropout rx_step max_grad_norm
  cutoff nconvs kw kh height mode lr pull pull_incr
  min_length batch_size grad_noise_scale task
  train_data_size init_weight curriculum_bound
  '''.split()

  def __init__(self, FLAGS, **kws):
    for key in self.config_keys:
      val = kws.get(key, getattr(FLAGS, key, None))
      setattr(self, key, val)

    min_length = 3
    max_length = min(FLAGS.max_length, data_utils.bins[-1])
    assert max_length + 1 > min_length
    self.max_length = max_length
    self.min_length = min_length

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

    self.min_length = model_config.min_length
    self.max_length = model_config.max_length
    self.model_config = model_config

  def is_valid_length(self, l):
    """Is this a valid length to pass in?"""
    return True

  def draw_length(self, cur_length):
    l = None
    while l is None:
      # Select the length for curriculum learning.
      l = np.random.randint(self.min_length, self.max_cur_length + 1)
      if np.random.randint(100) < 60: # Prefer longer stuff 60% of time.
        l = max(l, np.random.randint(self.min_length, cur_length + 1))
      # Mixed curriculum learning: in 25% of cases go to an even larger length.
      if np.random.randint(100) < 25:
        l = max(l, np.random.randint(self.min_length, self.max_length + 1))

      if not self.is_valid_length(l):
        l = None

    return l

  def test_examples(self, batch_size, task_name=None):
    generator = [g for g in self.generators if g.name == task_name][0]
    for l in np.arange(self.min_length, self.max_length + 1):
      if self.is_valid_length(l):
        yield (generator.get_batch(l, batch_size), l)

  def draw_example(self, batch_size, l=None, generator=None):
    """Draw a random example"""
    if generator is None:
      generator = self.draw_generator()
    if l is None:
      cur_length = self.get_cur_length(generator)
      l = self.draw_length(cur_length)
    return (generator.get_batch(l, batch_size), l)

  def tasks(self):
    """List of task names"""
    return [g.name for g in self.generators]

  def consider_extending(self, results):
    """Interpret the results"""
    pass

  def draw_generator(self):
    return np.random.choice(self.generators)

  def get_cur_length(self, generator):
    pass

class DefaultCurriculum(Curriculum):
  def __init__(self, generators, model_config):
    super(DefaultCurriculum, self).__init__(generators, model_config)

    self.max_cur_length = min(self.min_length + 3, self.max_length)

  def get_cur_length(self, generator):
    return self.max_cur_length

  def consider_extending(self, result):
    if result.avg_seq_err > self.model_config.curriculum_bound:
      return False
    if self.max_cur_length < self.max_length:
      self.max_cur_length += 1
      while not self.is_valid_length(self.max_cur_length) and self.max_cur_length < self.max_length:
        self.max_cur_length += 1

class MixedCurriculum(Curriculum):
  def __init__(self, generators, model_config):
    super(MixedCurriculum, self).__init__(generators, model_config)

  def get_cur_length(self, generator):
    pass

  def consider_extending(self, result):
    pass

class NeuralGPUResult(object):
  grad_norm = None
  back_update = None
  loss = None
  output = None
  step = None
  attention = None

  def __init__(self, vals, inp, target, taskid):
    self.__dict__.update(vals)
    self.input = inp
    self.target = target
    self.taskid = taskid

  def accuracy(self, nprint=0):
    batch_size = self.input.shape[1]
    return data_utils.accuracy(self.input, self.output, self.target, batch_size, nprint)

  @property
  def length(self):
    return (self.input[:,0] > 0).sum()

  def __repr__(self):
    err, tot, seq_err = self.accuracy()
    return '<NeuralGPUResult: length=%s loss=%s bs=%s err=%s seq_err=%s>' % \
      (self.length, self.loss, self.input.shape[1], err, seq_err)

class ResultsRecord(object):
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
    if below_curriculum:
      self.loss += results.loss
      err, tot, seq_err = results.accuracy()
      self.err += err
      self.seq_err += seq_err
      self.total += tot

  @property
  def avg_step_time(self):
    return self.step_time / self.num_batches

  @property
  def avg_grad_norm(self):
    return self.grad_norm / self.num_batches

  @property
  def avg_loss(self):
    return self.loss / self.num_below

  @property
  def avg_err(self):
    return self.err / self.total

  @property
  def avg_seq_err(self):
    return self.seq_err / (self.num_below * self.batch_size)
