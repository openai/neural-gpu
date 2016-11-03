"""Config object.

Ideally, all used FLAGS would be passed through here.  However, the
code is lazy in parts and uses FLAGS directly.
"""

from . import data_utils

class NeuralConfig(object):
  """Initial configuration settings for model"""

  config_keys = '''nmaps niclass noclass dropout rx_step max_grad_norm
  cutoff nconvs kw kh height mode lr pull pull_incr
  min_length batch_size task init_weight curriculum_bound layer_scale
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

  def __str__(self):
    msg1 = ("layers %d kw %d h %d kh %d relax %d batch %d task %s"
            % (self.nconvs, self.kw, self.height, self.kh, self.rx_step,
               self.batch_size, self.task))
    msg2 = ("cut %.2f pull %.3f lr %.2f iw %.2f cr %.2f nm %d d%.4f gn %.2f %s" %
            (self.cutoff, self.pull_incr, self.lr, self.init_weight,
             self.curriculum_bound, self.nmaps, self.dropout, self.max_grad_norm, msg1))
    return msg2

