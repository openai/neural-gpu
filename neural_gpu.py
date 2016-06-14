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

def conv_gru(mem, kw, kh, nmaps, cutoff, prefix):
  """Convolutional GRU."""
  def conv_lin(args, suffix, bias_start):
    return conv_linear(args, kw, kh, len(args) * nmaps, nmaps, True, bias_start,
                       prefix + "/" + suffix)
  reset = sigmoid_cutoff(conv_lin([mem], "r", 1.0), cutoff)
  candidate = tanh_cutoff(conv_lin([reset * mem], "c", 0.0), FLAGS.cutoff_tanh)
  # candidate = tf.tanh(conv_lin([reset * mem], "c", 0.0))
  gate = sigmoid_cutoff(conv_lin([mem], "g", 1.0), cutoff)
  return gate * mem + (1 - gate) * candidate

def gru_block(nconvs, cur, kw, kh, nmaps, cutoff, mask, suffix):
  # Do nconvs-many CGRU steps.
  for layer in xrange(nconvs):
    cur = conv_gru(cur, kw, kh, nmaps, cutoff, "cgru_%d_%s" % (layer, suffix))
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

def softmax(array):
  """Perform a softmax along the final axis but preserve shape."""
  nclass = array.get_shape()[-1].value
  result = tf.nn.softmax(tf.reshape(array, [-1, nclass]))
  return tf.reshape(result, [-1] + [x.value for x in array.get_shape()[1:]])

def check_nonzero(sparse):
  """In a sparse batch of ints, make 1 if it's > 0 and 0 else."""
  return tf.clip_by_value(sparse, 0, 1)

class NeuralGPUAtSize(object):
  """Instantiate the NeuralGPU at a given block size."""
  def __init__(self, model, length, adam):
    self.config = model.config
    self.length = length
    self.input = tf.concat(1, [tf.expand_dims(i, 1) for i in model.input[:length]])
    self.target = model.target
    #tf.concat(1, [tf.reshape(i, [-1, 1]) for i in model.target[:length]])
    self.emb_weights = model.emb_weights
    self.e0 = model.e0
    self.do_training = model.do_training

    self.model = model

    self.task = model.task

    self.construct_graph(adam)

  def construct_mask(self) :
    # Mask to 0-out padding space in each step.
    bmask = [(self.input[:,l] > 0) | (self.target[l] > 0) for l in xrange(self.length)]
    mask = [tf.to_float(tf.reshape(m, [-1, 1])) for m in bmask]
    # Use a shifted mask for step scaling and concatenated for weights.
    shifted_mask = mask + [tf.zeros_like(mask[0])]
    scales = [shifted_mask[i] * (1 - shifted_mask[i+1]) for i in xrange(self.length)]
    scales = [tf.reshape(s, [-1, 1, 1, 1]) for s in scales]
    mask = tf.concat(1, mask)  # batch x length
    mask = tf.reshape(mask, [-1, self.length, 1, 1])
    return mask, scales

  def construct_mask_better(self):
    # Mask to 0-out padding space in each step.
    # bmask: batch_size x length
    bmask = (self.input > 0) | (self.target > 0)
    # mask: batch_size x length x 1 x 1
    mask = tf.expand_dims(tf.expand_dims(bmask, 2), 2)
    return tf.to_float(mask)

  def construct_graph(self, adam):
    nmaps = self.config.nmaps
    vec_size = self.config.nmaps
    noclass = self.config.noclass
    cutoff = self.config.cutoff
    nconvs = self.config.nconvs
    kw = self.config.kw
    kh = self.config.kh
    height = self.config.height
    batch_size = tf.shape(self.input)[0]

    # The general tensor shape is
    # batchsize x length x height x nmaps

    # Embed inputs and calculate mask.
    if True:#with tf.device("/cpu:0"):
      with tf.control_dependencies([self.e0]):
        embedded = tf.nn.embedding_lookup(self.emb_weights, self.input)
      mask, scales = self.construct_mask()

    # start: batch_size x length x nmaps
    start = tf.tanh(embedded)

    # First image comes from start by applying one convolution and adding 0s.
    # first: batch_size x length x height x nmaps
    first = conv_linear(tf.expand_dims(start, 2),
                        1, 1, vec_size, nmaps, True, 0.0, "input")
    first = tf.concat(2, [first] + [tf.zeros_like(first)]*(height - 1))

    # Computation steps.
    keep_prob = 1.0 - self.do_training * (self.config.dropout * 8.0 / float(self.length))
    step = [tf.nn.dropout(first, keep_prob) * mask]
    curs = []
    self.attention_probs = []
    for it in xrange(self.length):
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
          import ipdb;ipdb.set_trace()
        else:
          cur = gru_block(nconvs, cur, kw, kh, nmaps, cutoff, mask, 'lookup')

        curs.append(cur)
        cur = tf.nn.dropout(cur, keep_prob)
        step.append(cur * mask)

    self.steps = [tf.reshape(s, [-1, self.length, height * nmaps]) for s in step]
    if FLAGS.do_lastout:
      # Final convolution to get logits, list outputs.
      outputs = []
      with tf.variable_scope("output") as vs:
        for i, layer in enumerate(curs):
          output = conv_linear(layer[:,:,:1,:], 1, 1, nmaps, noclass, True, 0.0, "o")
          #output = tf.reshape(output, [-1, self.length, noclass])
          outputs.append(output)
          vs.reuse_variables()

      # External outputs is length x batch_size x noclass
      # to match target
      external_outputs = [tf.transpose(softmax(tf.reshape(o, [-1, self.length, noclass])),
                                       [1,0,2]) for o in outputs]
      # external_output = [tf.nn.softmax(o) for o in external_output]
      # external_output[1] == character 1 for all batches
      #tf.transpose(tf.nn.softmax(tf.reshape(output, [-1, noclass])), [1,0,2])
      self.layer_outputs = external_outputs
      output = outputs[-1]
      self.output = external_outputs[-1]
    else:
      self.layer_outputs = []

      output = tf.add_n([curs[i][:,:,:1,:] * scales[i] for i in xrange(self.length)])
      output = conv_linear(output, 1, 1, nmaps, noclass, True, 0.0, "output")
      output = tf.reshape(output, [-1, self.length, noclass])
      external_output = [tf.reshape(o, [-1, noclass])
                         for o in list(tf.split(1, self.length, output))]
      external_output = [tf.nn.softmax(o) for o in external_output]

      self.output = external_output

    # Calculate cross-entropy loss and normalize it.
    targets = tf.concat(1, [make_dense(self.target[l], noclass)
                            for l in xrange(self.length)])
    targets = tf.reshape(targets, [-1, noclass])
    xent = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(
        tf.reshape(output, [-1, noclass]), targets), [-1, self.length])
    perp_loss = tf.reduce_sum(xent * tf.reshape(mask, [-1, self.length]))
    perp_loss /= tf.cast(batch_size, dtype=tf.float32)
    perp_loss /= self.length

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
    #import ipdb; ipdb.set_trace()

    def __repr__(self):
      return '<NeuralGPUAtSize %s>' % (self.length)

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

    # Feeds for inputs, targets, outputs, losses, etc.
    self.instances = []

    self.input = []
    self.target = []
    for l in xrange(data_utils.forward_max + 1):
      self.input.append(tf.placeholder(tf.int32, shape=(None,), name="inp{0}".format(l)))
      self.target.append(tf.placeholder(tf.int32, shape=(None,), name="tgt{0}".format(l)))
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
      self.instances.append(NeuralGPUAtSize(self, length, adam))
      tf.get_variable_scope().reuse_variables() # Later rounds reuse variables
      data_utils.print_out("Created model for bin of length %d in"
                           " %.2f s." % (length, time.time() - start_time))

  def get_instance_for_length(self, length):
    for instance in self.instances:
      if instance.length >= length:
        return instance
    raise IndexError('Max instance size %s; %s is too large!' % (instance.length, length))

  def step(self, sess, batch, do_backward, get_steps=False):
    """Run a step of the network."""
    inp, target, taskid = batch
    assert len(inp) == len(target)
    length = len(target)
    feed_in = {}
    feed_in[self.do_training] = 1.0 if do_backward else 0.0
    feed_in[self.task] = taskid
    feed_out = {}
    for l in xrange(length):
      feed_in[self.input[l]] = inp[l]
      feed_in[self.target[l]] = target[l]
    instance = self.get_instance_for_length(length)
    if do_backward:
      feed_out['back_update'] = instance.update
      feed_out['grad_norm'] = instance.grad_norm
    if get_steps:
      feed_out['step'] = instance.steps
    feed_out['loss'] = instance.loss
    feed_out['layer_outputs'] = instance.layer_outputs
    feed_out['output'] = instance.output
    if FLAGS.do_attention:
      feed_out['attention'] = instance.attention_probs
    res = sess.run(feed_out, feed_in)
    return neural_curriculum.NeuralGPUResult(res, inp, target, taskid)

