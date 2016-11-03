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

import random
import numpy as np

from . import mytf
from . import data_utils
from .records import NeuralGPUResult

FLAGS = tf.app.flags.FLAGS

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

def conv_linear(arg, kw, kh, nout, prefix, bias=0):
  """Convolutional linear map."""
  strides = [1, 1, 1, 1]
  if isinstance(arg, list):
    if len(arg) == 1:
      arg = arg[0]
    else:
      arg = tf.concat(len(mytf.shape_list(arg[0]))-1, arg)
  nin = mytf.shape_list(arg)[-1]
  with tf.variable_scope(prefix):
    k = tf.get_variable("CvK", [kw, kh, nin, nout])
    res = mytf.conv2d(arg, k, strides, "SAME")

    if bias is None:
      return res
    bias_term = tf.get_variable("CvB", [nout],
                                initializer=tf.constant_initializer(0.0))
    return res + bias_term + float(bias)

def conv_gru(mem, kw, kh, nmaps, cutoff, prefix, extras=[]):
  """Convolutional GRU."""
  # mem shape: bs x length x height x nmaps
  def conv_lin(arg, suffix, bias_start):
    return conv_linear(extras + [arg], kw, kh, nmaps,
                       prefix + "/" + suffix, bias=bias_start)
  reset = sigmoid_cutoff(conv_lin(mem, "r", 1), cutoff)
  candidate = tanh_cutoff(conv_lin(reset * mem, "c", 0), FLAGS.cutoff_tanh)
  gate = sigmoid_cutoff(conv_lin(mem, "g", 1), cutoff)
  return gate * mem + (1 - gate) * candidate

def resnet_block(cur, kw, kh, nmaps, cutoff, mask, suffix, nconvs=2,
                 extras = []):
  old = cur
  for i in range(nconvs):
    cur = conv_linear(extras + [cur], kw, kh, nmaps, "cgru_%d_%s" % (i, suffix))
    if i == nconvs - 1:
      cur = old + cur
    cur = tf.nn.relu(cur * mask)
  return cur

def lstm_block(cur, kw, kh, nmaps, cutoff, mask, suffix, nconvs=2,
               extras = []):
  # Do nconvs-many CGRU steps.
  for layer in range(nconvs):
    cur = conv_gru(cur, kw, kh, nmaps, cutoff, "cgru_%d_%s" % (layer, suffix),
                   extras = extras)
    cur *= mask
  return cur

def gru_block(*args, **kws):
  if FLAGS.do_resnet:
    return resnet_block(*args, **kws)
  else:
    return lstm_block(*args, **kws)

def relaxed_average(var_name_suffix, rx_step):
  """Calculate the average of relaxed variables having var_name_suffix."""
  relaxed_vars = []
  for l in range(rx_step):
    with tf.variable_scope("RX%d" % l, reuse=True):
      try:
        relaxed_vars.append(tf.get_variable(var_name_suffix))
      except ValueError:
        pass
  assert relaxed_vars
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

class NeuralGPUAtSize(object):
  """Instantiate the NeuralGPU at a given block size."""
  def __init__(self, model, length, adam):
    self.ntasks = 4

    self.config = model.config
    self.length = length
    # batch_size x length x height
    self.input = tf.placeholder(tf.int32, shape=(None, length, self.config.height),
                                name="input{0}".format(length))
    self.target = tf.placeholder(tf.int32, shape=(None,length), name="target{0}".format(length))
    self.emb_weights = model.emb_weights
    self.e0 = model.e0
    self.do_training = tf.placeholder(tf.bool, shape=[], name="do_training")

    self.model = model

    self.task = tf.placeholder(tf.uint8, shape=(None,), name="task")

    self.construct_graph(adam)

  def construct_mask(self) :
    # Mask to 0-out padding space in each step.
    # bmask: batch_size x length
    bmask = tf.reduce_any(self.input > 0, 2) | (self.target > 0)
    # mask: batch x length x 1 x 1
    mask = tf.to_float(mytf.expand_dims_by_k(bmask, 2))
    return mask

  def looping_layer(self, cur, index, *args):
    if FLAGS.output_layer == 1:
      output, = args
    keep_prob = 1.0 - tf.to_float(self.do_training) * (self.config.dropout * 8.0 / self.length)
    for it in range(self.config.rx_step):
      with tf.variable_scope("RX%d" % it) as vs:
        old = cur
        cur = tf.nn.dropout(cur, keep_prob)
        cur = gru_block(cur, self.config.kw, self.config.kh, self.config.nmaps,
                        self.config.cutoff, self.mask, 'lookup',
                        self.config.nconvs, extras=self.extras)

        if FLAGS.do_batchnorm:
          if FLAGS.do_batchnorm == 1:
            cur = mytf.batch_norm(cur, self.do_training, scope='bn')
          elif FLAGS.do_batchnorm == 2:
            cur = mytf.batch_norm(cur, self.do_training, self.mask, scope='bn')

        if FLAGS.output_layer == 1:
          output += cur
        else:
          cur = tf.select(tf.greater_equal(self.output_layers, index + it), cur, old)
    if FLAGS.output_layer == 1:
      return (cur, index + self.config.rx_step, output)
    else:
      return (cur, index + self.config.rx_step)

  def construct_all_layers(self, first, mask):
    # first: batch_size x length x height x nmaps

    output_layers = tf.to_int32(tf.reduce_sum(mask, [1,2,3]))

    cur = first
    layers = []

    extras = []
    if FLAGS.taskid:
      # bs x 1 x 1 x ntasks
      task = tf.one_hot(tf.to_int32(mytf.expand_dims_by_k(self.task, 2)),
                        self.ntasks)
      extras.append(mytf.broadcast_as(task, cur, [1,2]))

    self.mask = mask
    self.extras = extras
    self.output_layers = output_layers
    it = tf.get_variable("layer_index", [], dtype=tf.int32,
                         initializer=tf.constant_initializer(0))
    # Using swap is slower, but saves GPU memory.
    use_swap = bool(self.config.nmaps > 256 or (FLAGS.do_batchnorm and self.config.nmaps == 128))
    num_layers = int(self.config.layer_scale*self.length)
    args = [cur, it] + ([tf.zeros_like(cur)] if FLAGS.output_layer == 1 else [])
    result = tf.while_loop(lambda cur, it, *args: it < num_layers,
                            self.looping_layer,
                            args,
                            parallel_iterations=1,
                            swap_memory=use_swap)
    if FLAGS.output_layer == 1:
      ans = result[-1]
    else:
      ans = result[0]
    return ans

  def _get_first_layer(self, mask):
    """Turn the input into a batch_size x length x height x nmaps tensor"""
    nmaps = self.config.nmaps
    height = self.config.height

    # Embed inputs and calculate mask.
    with tf.control_dependencies([self.e0]):
      embedded = tf.nn.embedding_lookup(self.emb_weights, self.input)

    # first: batch_size x length x height x nmaps
    first = tf.tanh(embedded)

    return first

  def construct_graph(self, adam):
    nmaps = self.config.nmaps
    noclass = self.config.noclass

    mask = self.construct_mask()

    # The general tensor shape is
    # batchsize x length x height x nmaps
    first = self._get_first_layer(mask)

    # Computation steps.
    last_layer = self.construct_all_layers(first, mask)

    # Final convolution to get logits, list outputs.
    layer_output = conv_linear(last_layer[:,:,:1,:], 1, 1, noclass, "output")
    output = mytf.safe_squeeze(layer_output, -2) # batch x length x noclass

    self.output = mytf.softmax(output) # batch_size x length x noclass

    # Calculate cross-entropy loss and normalize it.
    targets = tf.one_hot(self.target, noclass)
    xent = mytf.softmax_cross_entropy_with_logits(output, targets) # shape: batch x length
    perp_loss = tf.reduce_mean(xent * tf.reshape(mask, [-1, self.length]))

    # Final loss: cross-entropy + shared parameter relaxation part.
    relax_dist, self.model.avg_op = relaxed_distance(self.config.rx_step)
    total_loss = perp_loss + relax_dist * self.model.pull
    self.loss = perp_loss

    # Gradients and Adam update operation.
    if self.length == data_utils.bins[0] or (self.config.mode == 0 and
                                        self.length < data_utils.bins[-1] + 1):
      data_utils.print_out("Creating backward for bin of length %d." % self.length)
      params = tf.trainable_variables()
      grads = tf.gradients(total_loss, params)
      grads, norm = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
      self.grad_norm = norm
      update = adam.apply_gradients(zip(grads, params),
                                    global_step=self.model.global_step)
      self.update = update

  def __repr__(self):
    return '<NeuralGPUAtSize %s>' % (self.length)

  def step(self, sess, batch, do_backward=False, get_steps=False, more_feed={}):
    """Run a step of the network."""
    inp, target, taskid = batch
    assert inp.shape[0] == target.shape[0]
    assert inp.shape[-1] == target.shape[1]
    if len(inp.shape) == 2:
      inp = np.expand_dims(inp, 1)
    assert len(inp.shape) == 3
    if inp.shape[1] < self.config.height:
      extra = self.config.height - inp.shape[1]
      inp = np.concatenate([inp] + [np.zeros_like(inp[:,:1,:])]*extra, axis=1)
    feed_in = {}
    feed_in[self.do_training] = do_backward
    feed_in[self.task] = taskid
    feed_in[self.input] = inp.transpose([0,2,1])
    feed_in[self.target] = target
    feed_out = {}
    feed_out.update(more_feed)
    if do_backward:
      feed_out['back_update'] = self.update
      feed_out['grad_norm'] = self.grad_norm
    if get_steps:
      feed_out['layers'] = self.layers
    if hasattr(self, 'probs'):
      feed_out['probs'] = self.probs
    feed_out['loss'] = self.loss
    feed_out['output'] = self.output
    res = sess.run(feed_out, feed_in)
    return NeuralGPUResult(res, inp, target, taskid)

class NeuralGPU(object):
  """Neural GPU Model."""
  def __init__(self, config):
    self.t = time.time()
    self.config = config

    # Feeds for parameters and ops to update them.
    self.global_step = tf.Variable(0, trainable=False)
    self.lr = tf.Variable(float(config.lr), trainable=False)
    self.lr_decay_op = self.lr.assign(self.lr * 0.98)
    self.pull = tf.Variable(float(config.pull), trainable=False)
    self.pull_incr_op = self.pull.assign(self.pull * config.pull_incr)

    # Feeds for inputs, targets, outputs, losses, etc.
    self.instances = []

    with tf.variable_scope("model") as vs:
      self.construct_graph()
      self.saver = tf.train.Saver(tf.all_variables())

  def construct_graph(self):
    vec_size = self.config.nmaps
    # Computation.
    self.emb_weights = tf.get_variable(
        "embedding", [self.config.niclass, vec_size],
        initializer=tf.random_uniform_initializer(-1.7, 1.7))
    self.e0 = tf.scatter_update(self.emb_weights,
                           tf.constant(0, dtype=tf.int32, shape=[1]),
                           tf.zeros([1, vec_size]))

    adam = tf.train.AdamOptimizer(self.lr, epsilon=1e-4, use_locking=True)

    # Main graph creation loop, for every bin in data_utils.
    for length in sorted(list(set(data_utils.bins + [data_utils.forward_max]))):
      data_utils.print_out("Creating model for bin of length %d." % length)
      start_time = time.time()
      self.instances.append(NeuralGPUAtSize(self, length, adam))
      tf.get_variable_scope().reuse_variables() # Later rounds reuse variables
      data_utils.print_out("Created model for bin of length %d in"
                           " %.2f s." % (length, time.time() - start_time))

  def get_instance_for_length(self, length):
    for instance in self.instances:
      if instance.length >= length:
        return instance
    raise IndexError('Max instance size %s; %s is too large!' % (instance.length, length))

  def step(self, sess, batch, *args, **kws):
    """Run a step of the network."""
    inp, target, taskid = batch
    instance = self.get_instance_for_length(target.shape[-1])
    return instance.step(sess, batch, *args, **kws)

  def simple_step(self, sess, a):
    """Run a simple operation on one input.

    Reverses the order for you, so you can input in little endian.
    """
    if isinstance(a, basestring):
      a = [data_utils.to_id(c) for c in a]
    else:
      a = list(a)
    l = self.get_instance_for_length(len(a)).length
    pad = l - len(a)
    input = np.array([a[::-1] + [0]*pad])
    result = self.step(sess, (input, input, [0]), False)
    relevant_output = result.output.argmax(axis=-1)[0, :(-pad if pad else None)]
    return ''.join(map(data_utils.to_symbol, relevant_output[::-1]))
