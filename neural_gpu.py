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
import mytf

import data_utils
import random
import numpy as np
import neural_curriculum

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

class VariableInitializer(object):

  def __init__(self):
    self.variable_usages = {}

  def record_variable(self, var, x, y):
    self.variable_usages.setdefault(var, [])
    self.variable_usages[var].append((x,y))

  def get_feed(self):
    ans = {}
    for k, lst in self.variable_usages.items():
      norms = [[tf.sqrt(tf.reduce_sum(v ** 2)) for v in pair] for pair in lst]
      ratios = [n1/n2 for n1, n2 in norms]
      ans[k] = ratios
    return ans

  def use_feed(self, sess, result):
    ops = []
    for k in self.variable_usages:
      ratio = np.median(result[k])
      #print 'initial scaling %s %s : %s' % (k.op.name, ratio, result[k])
      ops.append(tf.assign(k, k * ratio))
    sess.run(ops)

def conv_linear(arg, kw, kh, nin, nout, do_bias, bias_start, prefix, initializer):
  """Convolutional linear map."""
  with tf.variable_scope(prefix):
    k = tf.get_variable("CvK", [kw, kh, nin, nout])
    res = mytf.conv2d(arg, k, [1, 1, 1, 1], "SAME")
    initializer.record_variable(k, arg, res)

    if not do_bias: return res
    bias_term = tf.get_variable("CvB", [nout],
                                initializer=tf.constant_initializer(0.0))
    return res + bias_term + bias_start

def conv_gru(mem, kw, kh, nmaps, cutoff, prefix, initializer):
  """Convolutional GRU."""
  def conv_lin(arg, suffix, bias_start):
    return conv_linear(arg, kw, kh, nmaps, nmaps, True, bias_start,
                       prefix + "/" + suffix, initializer)
  reset = sigmoid_cutoff(conv_lin(mem, "r", 1.0), cutoff)
  candidate = tanh_cutoff(conv_lin(reset * mem, "c", 0.0), FLAGS.cutoff_tanh)
  gate = sigmoid_cutoff(conv_lin(mem, "g", 1.0), cutoff)
  return gate * mem + (1 - gate) * candidate

def resnet_block(cur, kw, kh, nmaps, cutoff, mask, suffix, initializer, nconvs=2):
  old = cur
  for i in xrange(nconvs):
    cur = conv_linear(cur, kw, kh, nmaps, nmaps, True, 0.0, "cgru_%d_%s" % (i, suffix),
                      initializer)
    if i == nconvs - 1:
      cur = old + cur
    cur = tf.nn.relu(cur * mask)
  return cur

def lstm_block(cur, kw, kh, nmaps, cutoff, mask, suffix, initializer, nconvs=2):
  # Do nconvs-many CGRU steps.
  for layer in xrange(nconvs):
    cur = conv_gru(cur, kw, kh, nmaps, cutoff, "cgru_%d_%s" % (layer, suffix),
                   initializer)
    cur *= mask
  return cur

def gru_block(*args):
  if FLAGS.do_resnet:
    return resnet_block(*args)
  else:
    return lstm_block(*args)

def relaxed_average(var_name_suffix, rx_step):
  """Calculate the average of relaxed variables having var_name_suffix."""
  relaxed_vars = []
  for l in xrange(rx_step):
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
    self.config = model.config
    self.length = length
    # batch_size x length
    self.input = tf.placeholder(tf.int32, shape=(None,self.config.input_height,length),
                                name="input{0}".format(length))
    self.target = tf.placeholder(tf.int32, shape=(None,length), name="target{0}".format(length))
    #tf.concat(1, [tf.reshape(i, [-1, 1]) for i in model.target[:length]])
    self.emb_weights = model.emb_weights
    self.e0 = model.e0
    self.do_training = tf.placeholder(tf.bool, shape=[], name="do_training")

    self.model = model

    self.task = tf.placeholder(tf.uint8, shape=(None,), name="task")

    if FLAGS.do_binarization:
      self.binary_gap = None

    self.initializer = VariableInitializer()
    self.construct_graph(adam)

  def construct_mask(self) :
    # Mask to 0-out padding space in each step.
    # bmask: batch_size x length
    bmask = tf.reduce_any(self.input > 0, 1) | (self.target > 0)
    # mask: batch x length x 1 x 1
    mask = tf.to_float(mytf.expand_dims_by_k(bmask, 2))

    # padded_mask: batch x (length+1) x 1 x 1
    padded_mask = tf.concat(1, [mask, tf.zeros_like(mask[:,:1,:,:])])
    # scales: initially batch x length x 1 x 1
    scales = padded_mask[:,:self.length,:,:] * (1 - padded_mask[:,1:,:,:])
    # Now length x batch x 1 x 1
    scales = tf.transpose(scales, [1,0,2,3])
    return mask, scales

  def construct_all_layers(self, first, mask):
    # first: batch_size x length x height x nmaps
    cutoff = self.config.cutoff
    kw = self.config.kw
    kh = self.config.kh
    nmaps = self.config.nmaps
    nconvs = self.config.nconvs

    avg_length = tf.reduce_mean(tf.reduce_sum(mask, [1,2,3])) # shape: batch_size
    #desired_norms = mytf.expand_dims_by_k(tf.sqrt(real_lengths), 3) / 6

    keep_prob = 1.0 - tf.to_float(self.do_training) * (self.config.dropout * 8.0 / self.length)
    cur = first
    layers = [first]
    attention_probs_list = []
    for it in xrange(self.length):
      with tf.variable_scope("RX%d" % (it % self.config.rx_step)) as vs:
        if it >= self.config.rx_step:
          vs.reuse_variables()
        cur = tf.nn.dropout(cur, keep_prob)

        if FLAGS.num_attention:
          k = FLAGS.num_attention
          blocks = tf.pack([cur]*(2*k+1))
          result = gru_block(blocks, kw, kh, nmaps, cutoff, mask, 'grublocks',
                             self.initializer, nconvs)
          # shape: (2k+1) x bs x length x height x nmaps
          keys = result[:k,:,:,:,:]
          vals = result[k:2*k,:,:,:,:]
          cur_att = result[2*k,:,:,:,:]
          logits = tf.reduce_sum(keys * cur_att, [-1,-2,-3]) # shape: k x bs
          attention_probs = tf.transpose(mytf.softmax(tf.transpose(logits))) # shape: k x bs
          attention_probs_list.append(attention_probs)
          cur = tf.reduce_sum(mytf.expand_dims_by_k(attention_probs, 3) * vals, 0)
        else:
          cur = gru_block(cur, kw, kh, nmaps, cutoff, mask, 'lookup',
                          self.initializer, nconvs)

        if FLAGS.do_binarization:
          dists_from_binary = 1 - tf.abs(2*cur - 1)
          total_dist = tf.reduce_sum(dists_from_binary, [-1,-2,-3])
          if self.binary_gap is None:
            self.binary_gap = total_dist
          else:
            self.binary_gap += total_dist

        if FLAGS.do_batchnorm:
          if FLAGS.do_batchnorm == 1:
            cur = mytf.batch_norm(cur, self.do_training, scope='bn')
          elif FLAGS.do_batchnorm == 2:
            cur = mytf.batch_norm(cur, self.do_training, mask, scope='bn')
          cur = cur * mask


        layers.append(cur)

    self.attention_probs = tf.pack(attention_probs_list) # shape: layers x 3 x bs
    self.layers = tf.pack(layers)
    return layers

  def _get_first_layer(self, mask):
    """Turn the input into a batch_size x length x height x nmaps tensor"""
    nmaps = self.config.nmaps
    vec_size = self.config.nmaps
    height = self.config.height

    # Embed inputs and calculate mask.
    with tf.control_dependencies([self.e0]):
      embedded = tf.nn.embedding_lookup(self.emb_weights, self.input)

    # start: batch_size x length x 1 x nmaps
    start = tf.tanh(tf.transpose(embedded, [0,2,1,3]))

    # First image comes from start by applying one convolution and adding 0s.
    # first: batch_size x length x height x nmaps
    if FLAGS.original_non_binary:
      first = conv_linear(start,
                          1, 1, vec_size, nmaps, True, 0.0, "input", self.initializer)
    else:
      first = start
    first = tf.concat(2, [first] + [tf.zeros_like(first[:,:,:1,:])]*(height - 1)) * mask

    return first

  def construct_graph(self, adam):
    nmaps = self.config.nmaps
    noclass = self.config.noclass

    mask, scales = self.construct_mask()

    # The general tensor shape is
    # batchsize x length x height x nmaps
    first = self._get_first_layer(mask)

    # Computation steps.
    self.construct_all_layers(first, mask)

    # Final convolution to get logits, list outputs.
    layer_output = conv_linear(self.layers[:,:,:,:1,:], 1, 1, nmaps, noclass, True, 0.0, "output", self.initializer)
    outputs = mytf.safe_squeeze(layer_output, -2) # (depth+1) x batch x length x noclass
    output = tf.reduce_sum(outputs[1:,:,:,:] * scales, 0)
    self.layer_outputs = mytf.softmax(outputs)
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

  def get_initial_scaling(self, sess, batch_size=32):
    feed = self.initializer.get_feed()
    batch = (np.random.randint(0, self.config.niclass, (batch_size, self.length)),
             np.random.randint(0, self.config.noclass, (batch_size, self.length)),
             np.random.randint(0, 1, batch_size))
    result = self.step(sess, batch, more_feed=feed)
    self.initializer.use_feed(sess, result.feed_out)

  def step(self, sess, batch, do_backward=False, get_steps=False, more_feed={}):
    """Run a step of the network."""
    inp, target, taskid = batch
    assert inp.shape[0] == target.shape[0]
    assert inp.shape[-1] == target.shape[-1]
    if len(inp.shape) == 2:
      inp = np.expand_dims(inp, 1)
    assert len(inp.shape) == 3
    if inp.shape[1] < self.config.input_height:
      extra = self.config.input_height - inp.shape[1]
      inp = np.concatenate([inp] + [np.zeros_like(inp[:,:1,:])]*extra, axis=1)
    feed_in = {}
    feed_in[self.do_training] = do_backward
    feed_in[self.task] = taskid
    feed_in[self.input] = inp
    feed_in[self.target] = target
    feed_out = {}
    feed_out.update(more_feed)
    if do_backward:
      feed_out['back_update'] = self.update
      feed_out['grad_norm'] = self.grad_norm
    if get_steps:
      feed_out['layers'] = self.layers
    feed_out['loss'] = self.loss
    feed_out['layer_outputs'] = self.layer_outputs
    feed_out['output'] = self.output
    feed_out['attention'] = self.attention_probs
    res = sess.run(feed_out, feed_in)
    return neural_curriculum.NeuralGPUResult(res, inp, target, taskid)

class NeuralGPU(object):
  """Neural GPU Model."""
  def __init__(self, config):
    self.t = time.time()
    self.config = config

    # Feeds for parameters and ops to update them.
    self.global_step = tf.Variable(0, trainable=False)
    self.lr = float(config.lr)

    self.pull = float(config.pull)

    # Feeds for inputs, targets, outputs, losses, etc.
    self.instances = []

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

  def renormalize(self, sess):
    self.instances[0].get_initial_scaling(sess)

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
