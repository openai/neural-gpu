# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The Neural GPU Model."""

import time

import tensorflow as tf

import data_utils
import random
import numpy as np

FLAGS = tf.app.flags.FLAGS

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

    self.binary_annealing = FLAGS.binary_activation
    self.do_binary = FLAGS.binary_activation >= 0

  def __str__(self):
    msg1 = ("layers %d kw %d h %d kh %d relax %d batch %d noise %.2f task %s"
            % (self.nconvs, self.kw, self.height, self.kh, self.rx_step,
               self.batch_size, self.grad_noise_scale, self.task))
    msg2 = "data %d %s" % (self.train_data_size, msg1)
    msg3 = ("cut %.2f pull %.3f lr %.2f iw %.2f cr %.2f nm %d d%.4f gn %.2f %s" %
            (self.cutoff, self.pull_incr, self.lr, self.init_weight,
            self.curriculum_bound, self.nmaps, self.dropout, self.max_grad_norm, msg2))
    return msg3

class DefaultCurriculum(object):
  def __init__(self, generators, model_config):
    self.generators = generators
    self.model_config = model_config

    self.min_length = model_config.min_length
    self.max_length = model_config.max_length
    self.max_cur_length = min(self.min_length + 3, self.max_length)

    self.prev_acc_perp = None
    self.prev_seq_err = None

  def tasks(self):
    return [g.name for g in self.generators]

  def is_valid_length(self, l):
    return any(g.is_valid_length(l) for g in self.generators)

  def test_examples(self, batch_size, task_name):
    for g in self.generators:
      if g.name == task_name:
        break
    else:
      raise KeyError("No such task")
    for l in np.arange(self.min_length, self.max_length + 1):
      yield (l, self.draw_example(batch_size, l)[0])

  def draw_example(self, batch_size, l=None):
    generator = random.choice(self.generators)

    if l is None:
      # Select the length for curriculum learning.
      l = np.random.randint(self.min_length, self.max_cur_length + 1)
      if np.random.randint(100) < 60: # Prefer longer stuff 60% of time.
        l = max(l, np.random.randint(self.min_length, self.max_cur_length + 1))
      # Mixed curriculum learning: in 25% of cases go to an even larger length.
      if np.random.randint(100) < 25:
        l = max(l, np.random.randint(self.min_length, self.max_length + 1))

    result = generator.get_batch(l, batch_size)
    return (result, l)

  def consider_extending(self, acc):
    if acc > self.model_config.curriculum_bound:
      return False
    if self.max_cur_length < self.max_length:
      self.max_cur_length += 1
      while not self.is_valid_length(self.max_cur_length) and self.max_cur_length < self.max_length:
        self.max_cur_length += 1


def conv_linear(args, kw, kh, nin, nout, do_bias, bias_start, prefix):
  """Convolutional linear map."""
  assert args is not None
  if not isinstance(args, (list, tuple)):
    args = [args]
  with tf.variable_scope(prefix):
    k = tf.get_variable("CvK", [kw, kh, nin, nout])
    if len(args) == 1:
      res = tf.nn.conv2d(args[0], k, [1, 1, 1, 1], "SAME")
    else:
      res = tf.nn.conv2d(tf.concat(3, args), k, [1, 1, 1, 1], "SAME")
    if not do_bias: return res
    bias_term = tf.get_variable("CvB", [nout],
                                initializer=tf.constant_initializer(0.0))
    return res + bias_term + bias_start


def tf_cut_function(val, vlo, vhi, glo, ghi):
  if vlo is None:
    return val
  a = tf.clip_by_value(val, vlo, vhi)
  if glo is None:
    return a
  assert ghi >= vhi > vlo >= glo
  zz = tf.clip_by_value(val, glo, ghi)
  return zz - tf.stop_gradient(zz - a)

def sigmoid_cutoff(x, cutoff):
  """Sigmoid with cutoff, e.g., 1.2sigmoid(x) - 0.1."""
  y = tf.sigmoid(x)
  if cutoff < 1.01: return y
  d = (cutoff - 1.0) / 2.0
  z = cutoff * y - d
  dd = (FLAGS.smooth_grad - 1.0) / 2.0 if FLAGS.smooth_grad else None
  glo, ghi = (-dd, 1+dd) if FLAGS.smooth_grad else (None, None)
  return tf_cut_function(z, 0, 1, glo, ghi)


def tanh_cutoff(x, cutoff):
  """Tanh with cutoff, e.g., 1.1tanh(x) cut to [-1. 1]."""
  y = tf.tanh(x)
  if cutoff < 1.01: return y
  z = cutoff * y
  tcut = FLAGS.smooth_grad_tanh
  glo, ghi = (-tcut, tcut) if tcut else (None, None)
  return tf_cut_function(z, -1, 1, glo, ghi)

def conv_gru(inpts, mem, kw, kh, nmaps, cutoff, prefix):
  """Convolutional GRU."""
  def conv_lin(args, suffix, bias_start):
    return conv_linear(args, kw, kh, len(args) * nmaps, nmaps, True, bias_start,
                       prefix + "/" + suffix)
  reset = sigmoid_cutoff(conv_lin(inpts + [mem], "r", 1.0), cutoff)
  candidate = tanh_cutoff(conv_lin(inpts + [reset * mem], "c", 0.0), FLAGS.cutoff_tanh)
  # candidate = tf.tanh(conv_lin(inpts + [reset * mem], "c", 0.0))
  gate = sigmoid_cutoff(conv_lin(inpts + [mem], "g", 1.0), cutoff)
  return gate * mem + (1 - gate) * candidate

def gru_block(nconvs, cur, kw, kh, nmaps, cutoff, mask, suffix):
  # Do nconvs-many CGRU steps.
  for layer in xrange(nconvs):
    cur = conv_gru([], cur, kw, kh, nmaps, cutoff, "cgru_%d_%s" % (layer, suffix))
    cur *= mask
  return cur

try:
  @tf.RegisterGradient("CustomIdG")
  def _custom_id_grad(_, grads):
    return grads
except KeyError as e: # Happens on reload
  pass

def quantize(t, quant_scale, max_value=1.0):
  """Quantize a tensor t with each element in [-max_value, max_value]."""
  t = tf.minimum(max_value, tf.maximum(t, -max_value))
  big = quant_scale * (t + max_value) + 0.5
  with tf.get_default_graph().gradient_override_map({"Floor": "CustomIdG"}):
    res = (tf.floor(big) / quant_scale) - max_value
  return res


def quantize_weights_op(quant_scale, max_value):
  ops = [v.assign(quantize(v, quant_scale, float(max_value)))
         for v in tf.trainable_variables()]
  return tf.group(*ops)


def relaxed_average(var_name_suffix, rx_step):
  """Calculate the average of relaxed variables having var_name_suffix."""
  relaxed_vars = []
  for l in xrange(rx_step):
    with tf.variable_scope("RX%d" % l, reuse=True):
      try:
        relaxed_vars.append(tf.get_variable(var_name_suffix))
      except ValueError:
        pass
  dsum = tf.add_n(relaxed_vars)
  avg = dsum / len(relaxed_vars)
  diff = [v - avg for v in relaxed_vars]
  davg = tf.add_n([d*d for d in diff])
  return avg, tf.reduce_sum(davg)


def relaxed_distance(rx_step):
  """Distance between relaxed variables and their average."""
  res, ops, rx_done = [], [], {}
  for v in tf.trainable_variables():
    vals = v.op.name.split('/', 2)
    if vals[1].startswith('RX'):
      rx_name = vals[2]
      if rx_name not in rx_done:
        avg, dist_loss = relaxed_average(rx_name, rx_step)
        res.append(dist_loss)
        rx_done[rx_name] = avg
      ops.append(v.assign(rx_done[rx_name]))
  return tf.add_n(res), tf.group(*ops)


def make_dense(targets, noclass):
  """Move a batch of targets to a dense 1-hot representation."""
  if True:#with tf.device("/cpu:0"):
    shape = tf.shape(targets)
    batch_size = shape[0]
    indices = targets + noclass * tf.range(0, batch_size)
    length = tf.expand_dims(batch_size * noclass, 0)
    dense = tf.sparse_to_dense(indices, length, 1.0, 0.0)
  return tf.reshape(dense, [-1, noclass])


def check_for_zero(sparse):
  """In a sparse batch of ints, make 1.0 if it's 0 and 0.0 else."""
  return 1.0-tf.to_float(tf.clip_by_value(sparse, 0, 1))
  if True:#with tf.device("/cpu:0"):
    shape = tf.shape(sparse)
    batch_size = shape[0]
    sparse = tf.minimum(sparse, 1)
    indices = sparse + 2 * tf.range(batch_size)
    dense = tf.sparse_to_dense(indices, tf.expand_dims(2 * batch_size, 0),
                               1.0, 0.0)
    reshaped = tf.reshape(dense, [-1, 2])
  return tf.reshape(tf.slice(reshaped, [0, 0], [-1, 1]), [-1])


class NeuralGPU(object):
  """Neural GPU Model."""

  def __init__(self, config):
    self.t = time.time()
    self.config = config

    # Feeds for parameters and ops to update them.
    self.global_step = tf.Variable(0, trainable=False)
    self.lr = float(config.lr)
    self.quant_op = quantize_weights_op(512, 8)

    self.pull = float(config.pull)
    self.do_training = tf.placeholder(tf.float32, name="do_training")

    self.binary_activation = 1.0

    # Feeds for inputs, targets, outputs, losses, etc.
    self.input = []
    self.target = []
    for l in xrange(data_utils.forward_max + 1):
      self.input.append(tf.placeholder(tf.int32, name="inp{0}".format(l)))
      self.target.append(tf.placeholder(tf.int32, name="tgt{0}".format(l)))
    self.outputs = []
    self.losses = []
    self.grad_norms = []
    self.updates = []
    self.task = tf.placeholder(tf.uint8, shape=(None,), name="task")

    with tf.variable_scope("model") as vs:
      self.construct_graph()
      self.saver = tf.train.Saver(tf.all_variables())

  def construct_graph(self):
    vec_size = self.config.nmaps
    # Computation.
    if True:#with tf.device("/cpu:0"):
      self.emb_weights = tf.get_variable(
          "embedding", [self.config.niclass, vec_size],
          initializer=tf.random_uniform_initializer(-1.7, 1.7))
      self.e0 = tf.scatter_update(self.emb_weights,
                             tf.constant(0, dtype=tf.int32, shape=[1]),
                             tf.zeros([1, vec_size]))

    adam = tf.train.AdamOptimizer(self.lr, epsilon=1e-4, use_locking=True)

    # Main graph creation loop, for every bin in data_utils.
    self.steps = []
    for length in sorted(list(set(data_utils.bins + [data_utils.forward_max]))):
      data_utils.print_out("Creating model for bin of length %d." % length)
      start_time = time.time()
      self.construct_graph_for_length(length, adam)
      tf.get_variable_scope().reuse_variables() # Later rounds reuse variables
      data_utils.print_out("Created model for bin of length %d in"
                           " %.2f s." % (length, time.time() - start_time))

  def construct_graph_for_length(self, length, adam):
    nmaps = self.config.nmaps
    vec_size = self.config.nmaps
    noclass = self.config.noclass
    cutoff = self.config.cutoff
    nconvs = self.config.nconvs
    kw = self.config.kw
    kh = self.config.kh
    height = self.config.height
    batch_size = tf.shape(self.input[0])[0]


    # Embed inputs and calculate mask.
    if True:#with tf.device("/cpu:0"):
      with tf.control_dependencies([self.e0]):
        embedded = [tf.nn.embedding_lookup(self.emb_weights, self.input[l])
                    for l in xrange(length)]
      # Mask to 0-out padding space in each step.
      imask = [check_for_zero(self.input[l]) for l in xrange(length)]
      omask = [check_for_zero(self.target[l]) for l in xrange(length)]
      mask = [1.0 - (imask[i] * omask[i]) for i in xrange(length)]
      mask = [tf.reshape(m, [-1, 1]) for m in mask]
      # Use a shifted mask for step scaling and concatenated for weights.
      shifted_mask = mask + [tf.zeros_like(mask[0])]
      scales = [shifted_mask[i] * (1.0 - shifted_mask[i+1])
                for i in xrange(length)]
      scales = [tf.reshape(s, [-1, 1, 1, 1]) for s in scales]
      mask = tf.concat(1, mask[0:length])  # batch x length
      weights = mask
      # Add a height dimension to mask to use later for masking.
      mask = tf.reshape(mask, [-1, length, 1, 1])
      mask = tf.concat(2, [mask for _ in xrange(height)]) + tf.zeros(
          tf.pack([batch_size, length, height, nmaps]), dtype=tf.float32)

    # Start is a length-list of batch-by-nmaps tensors, reshape and concat.
    start = [tf.tanh(embedded[l]) for l in xrange(length)]
    start = [tf.reshape(start[l], [-1, 1, nmaps]) for l in xrange(length)]
    start = tf.reshape(tf.concat(1, start), [-1, length, 1, nmaps])

    # First image comes from start by applying one convolution and adding 0s.
    first = conv_linear(start, 1, 1, vec_size, nmaps, True, 0.0, "input")
    first = [first] + [tf.zeros(tf.pack([batch_size, length, 1, nmaps]),
                                dtype=tf.float32) for _ in xrange(height - 1)]
    first = tf.concat(2, first)

    # Computation steps.
    keep_prob = 1.0 - self.do_training * (self.config.dropout * 8.0 / float(length))
    step = [tf.nn.dropout(first, keep_prob) * mask]
    outputs = []
    self.attention_probs = []
    for it in xrange(length):
      with tf.variable_scope("RX%d" % (it % self.config.rx_step)) as vs:
        if it >= self.config.rx_step:
          vs.reuse_variables()
        cur = step[it]


        if FLAGS.do_attention:
          cur_att = gru_block(nconvs, cur, kw, kh, nmaps, cutoff, mask, 'lookup')
          attention_vals = []
          logit_table = []
          for i in range(3):
            key = gru_block(nconvs, cur, kw, kh, nmaps, cutoff, mask, 'key%s' % i)
            val = gru_block(nconvs, cur, kw, kh, nmaps, cutoff, mask, 'val%s' % i)
            if i in [0,1]:
              val = tf.select(tf.equal(self.task, i), val, tf.stop_gradient(val))
              key = tf.select(tf.equal(self.task, i), key, tf.stop_gradient(key))
            logit = tf.reduce_sum(cur_att * key, [1,2,3])
            logit_table.append(tf.expand_dims(logit, 1))
            attention_vals.append(tf.expand_dims(val, 0))

          attention_probs = tf.transpose(tf.nn.softmax(tf.concat(1, logit_table)))
          self.attention_probs.append(attention_probs)
          attention_vals = tf.concat(0, attention_vals)
          expanded_probs = attention_probs # add 3 more dimensions
          for i in range(3):
            expanded_probs = tf.expand_dims(expanded_probs, -1)
          cur = tf.reduce_sum(expanded_probs * attention_vals, [0])
        else:
          cur = gru_block(nconvs, cur, kw, kh, nmaps, cutoff, mask, 'lookup')

        if self.config.do_binary:
          binary_cur = tf.sign(cur) - tf.stop_gradient(tf.sign(cur) - cur)
          cur = self.binary_activation * cur + (1 - self.binary_activation) * binary_cur

        outputs.append(tf.slice(cur, [0, 0, 0, 0], [-1, -1, 1, -1]))
        cur = tf.nn.dropout(cur, keep_prob)
        step.append(cur * mask)

    self.steps.append([tf.reshape(s, [-1, length, height * nmaps])
                       for s in step])
    # Output is the n-th step output; n = current length, as in scales.
    output = tf.add_n([outputs[i] * scales[i] for i in xrange(length)])
    # Final convolution to get logits, list outputs.
    output = conv_linear(output, 1, 1, nmaps, noclass, True, 0.0, "output")
    output = tf.reshape(output, [-1, length, noclass])
    external_output = [tf.reshape(o, [-1, noclass])
                       for o in list(tf.split(1, length, output))]
    external_output = [tf.nn.softmax(o) for o in external_output]
    # external_output[1] == character 1 for all batches
    #tf.transpose(tf.nn.softmax(tf.reshape(output, [-1, noclass])), [1,0,2])
    self.outputs.append(external_output)
    # Calculate cross-entropy loss and normalize it.
    targets = tf.concat(1, [make_dense(self.target[l], noclass)
                            for l in xrange(length)])
    targets = tf.reshape(targets, [-1, noclass])
    real_targets = ((0.5*targets + 0.5*tf.stop_gradient(
      tf.nn.softmax(tf.reshape(output, [-1, noclass])))) if
                    FLAGS.smooth_targets else targets)
    xent = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(
        tf.reshape(output, [-1, noclass]), real_targets), [-1, length])
    perp_loss = tf.reduce_sum(xent * weights)
    perp_loss /= tf.cast(batch_size, dtype=tf.float32)
    perp_loss /= length

    # Final loss: cross-entropy + shared parameter relaxation part.
    relax_dist, self.avg_op = relaxed_distance(self.config.rx_step)
    total_loss = perp_loss + relax_dist * self.pull
    self.losses.append(perp_loss)

    # Gradients and Adam update operation.
    if length == data_utils.bins[0] or (self.config.mode == 0 and
                                        length < data_utils.bins[-1] + 1):
      data_utils.print_out("Creating backward for bin of length %d." % length)
      params = tf.trainable_variables()
      grads = tf.gradients(total_loss, params)
      grads, norm = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
      self.grad_norms.append(norm)
      update = adam.apply_gradients(zip(grads, params),
                                    global_step=self.global_step)
      self.updates.append(update)


  def step(self, sess, batch, do_backward, get_steps=False):
    """Run a step of the network."""
    inp, target, taskid = batch
    assert len(inp) == len(target)
    length = len(target)
    feed_in = {}
    feed_in[self.do_training] = 1.0 if do_backward else 0.0
    feed_in[self.task] = taskid
    feed_out = {}
    index = len(data_utils.bins)
    if length < data_utils.bins[-1] + 1:
      index = data_utils.bins.index(length)
    for l in xrange(length):
      feed_in[self.input[l]] = inp[l]
      feed_in[self.target[l]] = target[l]
    if index >= len(self.losses):
      raise IndexError('index too large!')
    if do_backward:
      feed_out['back_update'] = self.updates[index]
      feed_out['grad_norm'] = self.grad_norms[index]
    if get_steps:
      feed_out['step'] = self.steps[index][:length+1]
    feed_out['loss'] = self.losses[index]
    feed_out['output'] = self.outputs[index][:length]
    if FLAGS.do_attention:
      feed_out['attention'] = self.attention_probs
    res = data_utils.sess_run_dict(sess, feed_out, feed_in)
    return NeuralGPUResult(res, inp, target, taskid)

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
