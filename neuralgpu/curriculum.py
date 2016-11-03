"""Curriculum and its subclasses decide when to choose which task for training."""

from __future__ import print_function

import numpy as np

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
      within_bounds = True
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
  last_if_solved = False

  def __init__(self, generators, model_config, kind):
    super(BetterCurriculum, self).__init__(generators, model_config)
    if kind == 2:
      self.decrease_threshold = 0.01
    elif kind == 3:
      self.rand_prob = 0
    elif kind == 4:
      self.only_later = True
    elif kind == 5:
      self.only_later = True
      self.last_if_solved = True

  def draw_generator(self, task=None):
    if task is not None:
      return [g for g in self.generators if g.name == task][0]
    unsolved = [g for g in self.generators if self.max_cur_lengths[g.taskid] < self.max_length]
    if not unsolved:
      if self.last_if_solved:
        return self.generators[-1]
      else:
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
