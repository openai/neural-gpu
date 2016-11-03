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
"""Neural GPU for Learning Algorithms."""

from __future__ import print_function

import math
import os
import random
import sys
import time
import subprocess
import yaml

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

from . import data_utils as data
from .generators import generators
from .model import NeuralGPU
from . import curriculum
from . import mytf
from . import records
from .config import NeuralConfig

def define_flags():
  """This is placed in a function so reload() works"""
  tf.app.flags.DEFINE_float("lr", 0.001, "Learning rate.")
  tf.app.flags.DEFINE_float("init_weight", 1.0, "Initial weights deviation.")
  tf.app.flags.DEFINE_float("max_grad_norm", 1.0, "Clip gradients to this norm.")
  tf.app.flags.DEFINE_float("cutoff", 1.2, "Cutoff at the gates.")
  tf.app.flags.DEFINE_float("cutoff_tanh", 0.0, "Cutoff at tanh.")
  tf.app.flags.DEFINE_float("pull", 0.0005, "Starting pull of the relaxations.")
  tf.app.flags.DEFINE_float("pull_incr", 1.2, "Increase pull by that much.")
  tf.app.flags.DEFINE_float("curriculum_bound", 0.15, "Move curriculum < this.")
  tf.app.flags.DEFINE_float("dropout", 0.15, "Dropout that much.")
  tf.app.flags.DEFINE_integer("max_steps", 0, "Quit after this many steps.")
  tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size.")
  tf.app.flags.DEFINE_integer("low_batch_size", 16, "Low batch size.")
  tf.app.flags.DEFINE_integer("steps_per_epoch", 200, "Steps per epoch.")
  tf.app.flags.DEFINE_integer("nmaps", 24, "Number of floats in each cell.")
  tf.app.flags.DEFINE_integer("niclass", 33, "Number of classes (0 is padding).")
  tf.app.flags.DEFINE_integer("noclass", 33, "Number of classes (0 is padding).")
  tf.app.flags.DEFINE_integer("max_length", 41, "Maximum length.")
  tf.app.flags.DEFINE_integer("rx_step", 6, "Relax that many recursive steps.")
  tf.app.flags.DEFINE_integer("random_seed", 125459, "Random seed.")
  tf.app.flags.DEFINE_integer("time_till_ckpt", 30, "How many tests per checkpoint")
  tf.app.flags.DEFINE_integer("time_till_eval", 2, "Number of steps between evals")
  tf.app.flags.DEFINE_integer("nconvs", 2, "How many convolutions / 1 step.")
  tf.app.flags.DEFINE_integer("kw", 3, "Kernel width.")
  tf.app.flags.DEFINE_integer("kh", 3, "Kernel height.")
  tf.app.flags.DEFINE_integer("height", 4, "Height.")
  tf.app.flags.DEFINE_integer("forward_max", 401, "Maximum forward length.")
  tf.app.flags.DEFINE_integer("nprint", 0, "How many test examples to print out.")
  tf.app.flags.DEFINE_integer("mode", 0, "Mode: 0-train other-decode.")
  tf.app.flags.DEFINE_bool("animate", False, "Whether to produce an animation.")
  tf.app.flags.DEFINE_float("smooth_grad", 0.0, "Whether to avoid clipping gradient")
  tf.app.flags.DEFINE_float("smooth_grad_tanh", 0.0, "Whether to avoid clipping tanh gradient")
  tf.app.flags.DEFINE_string("task", "badd", "Which task are we learning?")
  tf.app.flags.DEFINE_string("train_dir", "/tmp/neural", "Directory to store models.")

  tf.app.flags.DEFINE_float("layer_scale", 1.0, "Number of layers to use")

  # Batchnorm:     0 = none
  #                2 = correct
  #                1 = not quite correct, because of how masking is done, but simpler.
  tf.app.flags.DEFINE_integer("do_batchnorm", 0, "Whether to use batch normalization.")

  tf.app.flags.DEFINE_bool("do_resnet", False, "Whether to use resnets.")

  tf.app.flags.DEFINE_bool("print_one", True, "Print one example each evaluation")

  # output layer: 0 = standard: output layer n on length-n inputs
  #               1 = alternate: output sum of first n layers on length-n inputs.
  tf.app.flags.DEFINE_integer("output_layer", 0, "Which layer to output.")

  # progressive_curriculum: 0 = none: always train on first task.
  #                         1-5: progress through the tasks in sequence,
  #                              training each one to length max_len then move on.
  #                              The different options have subtle changes; see
  #                              BetterCurriculum for details.
  #                              5 is probably the best one.
  tf.app.flags.DEFINE_integer("progressive_curriculum", 0, "Whether to use progressive curriculum.")
  tf.app.flags.DEFINE_bool("taskid", False, "Feed task id to algorithm in each layer")

  tf.app.flags.DEFINE_bool("always_large", False, "Perform the large test even when the model is inaccurate")

FLAGS = tf.app.flags.FLAGS
if not FLAGS.__parsed: # Hack so reload() works
  define_flags()

EXTRA_EVAL = 2


log_output = None
step_output = None

def log_parameters(checkpoint_dir):
  """Write enough information in checkpoint_dir for reproducibility.

  Also check that we're in a new checkpoint directory.
  """
  global log_output, step_output
  command_fname = os.path.join(checkpoint_dir, 'commandline')
  if gfile.Exists(command_fname):
    old_argv = open(command_fname).read().strip()
    new_argv = ' '.join(sys.argv)
    if old_argv != new_argv:
      data.print_out('ERROR: restarted with changed argv')
      data.print_out('WAS %s' % old_argv)
      data.print_out('NOW %s' % new_argv)
      raise ValueError("Bad log dir: partial state exists with different arguments")
    else:
      print('Reusing existing log dir')
      #raise ValueError("Even though the argv didn't change, we'll still kill you.")

  with open(command_fname, 'w') as f:
    f.write(' '.join(sys.argv)+'\n')

  with open(os.path.join(checkpoint_dir, 'all_args'), 'w') as f:
    yaml.dump(FLAGS.__flags, f, default_flow_style=False)

  with open(os.path.join(checkpoint_dir, 'git-rev'), 'w') as f:
    subprocess.call(['git', 'rev-parse', 'HEAD'], stdout=f)

  log_output = open(os.path.join(checkpoint_dir, 'results'), 'a', 1)
  step_output = open(os.path.join(checkpoint_dir, 'steps'), 'a', 1)

def load_model(sess, checkpoint_dir, reconfig={}):
  # possibly tf.reset_default_graph()
  with open(os.path.join(checkpoint_dir, 'all_args')) as f:
    options = yaml.load(f)
  options.update(reconfig)
  FLAGS._parse_flags()
  FLAGS.__flags.update(options)
  data.forward_max = max(FLAGS.forward_max, data.bins[-1])
  config = NeuralConfig(FLAGS)
  model = NeuralGPU(config)
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    model.saver.restore(sess, ckpt.model_checkpoint_path)
  return model

def get_checkpoint_dir():
  #return FLAGS.train_dir + ('-seed%s-pid%s' % (FLAGS.random_seed, os.getpid()))
  return FLAGS.train_dir

def get_config_from_flags(checkpoint_dir = None):
  # Set random seed.
  seed = FLAGS.random_seed
  tf.set_random_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  # Create checkpoint directory if it does not exist.
  if checkpoint_dir is None:
    checkpoint_dir = get_checkpoint_dir()
  if not gfile.IsDirectory(checkpoint_dir):
    data.print_out("Creating checkpoint directory %s." % checkpoint_dir)
    try:
      gfile.MkDir(os.path.dirname(checkpoint_dir))
    except OSError as e:
      pass
    gfile.MkDir(checkpoint_dir)

  data.err_tee = data.TeeErr(open(os.path.join(checkpoint_dir, "err"), 'w'))

  data.print_out("NN ", newline=False)

  config = NeuralConfig(FLAGS)

  # Check data sizes.
  while len(data.bins) > 1 and data.bins[-2] > config.max_length + EXTRA_EVAL:
    data.bins = data.bins[:-1]
  assert data.bins[0] > FLAGS.rx_step
  data.forward_max = max(FLAGS.forward_max, data.bins[-1])

  return config

def initialize(sess, checkpoint_dir=None):
  """Initialize data and model."""
  config = get_config_from_flags(checkpoint_dir)
  data.print_out(str(sys.argv))
  data.print_out(str(config))

  if checkpoint_dir is None:
    checkpoint_dir = get_checkpoint_dir()
  log_parameters(checkpoint_dir)

  # Initialize data for each task.
  nclass = min(config.niclass, config.noclass)
  tasks = config.task.split(",")
  data_generators = [generators[t] for t in tasks]
  for g in data_generators:
    g._initialize(nclass)

  # Create model and initialize it.
  tf.get_variable_scope().set_initializer(
      tf.uniform_unit_scaling_initializer(factor=1.8 * FLAGS.init_weight))
  model = NeuralGPU(config)
  data.print_out("Created model.")
  sess.run(tf.initialize_all_variables())
  data.print_out("Initialized variables.")

  # Load model from parameters if a checkpoint exists.
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  model.curriculum = None
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    data.print_out("Reading model parameters from %s"
                   % ckpt.model_checkpoint_path)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    try:
      model.curriculum = yaml.load(open(os.path.join(checkpoint_dir, 'neural_gpu_curriculum.ckpt')))
    except IOError:
      pass

  if model.curriculum is None:
    if FLAGS.progressive_curriculum:
      model.curriculum = curriculum.BetterCurriculum(data_generators, model.config,
                                                      FLAGS.progressive_curriculum)
    else:
      model.curriculum = curriculum.GeneralizeCurriculum(data_generators, model.config)

  # Return the model and needed variables.
  return model


def single_test(l, model, sess, task, nprint, batch_size, print_out=True,
                offset=None, get_steps=False, batch=None):
  """Test model on test data of length l using the given session."""
  if batch is None:
    batch, _ = model.curriculum.draw_example(batch_size, l, task)
  result = model.step(sess, batch, False, get_steps=get_steps)
  errors, total, seq_err = result.accuracy(nprint)
  seq_err = float(seq_err) / batch_size
  if total > 0:
    errors = float(errors) / total
  if print_out:
    data.print_out("  %s len %d errors %.2f sequence-errors %.2f"
                   % (task, l, 100*errors, 100*seq_err))
  return errors, seq_err, result


def multi_test(l, model, sess, task, nprint, batch_size, offset=None):
  """Run multiple tests at lower batch size to save memory."""
  errors, seq_err = 0.0, 0.0
  to_print = nprint
  low_batch = FLAGS.low_batch_size
  low_batch = min(low_batch, batch_size)
  for mstep in range(batch_size // low_batch):
    cur_offset = None if offset is None else offset + mstep * low_batch
    err, sq_err, result = single_test(l, model, sess, task, to_print, low_batch,
                                 False, cur_offset)
    to_print = max(0, to_print - low_batch)
    errors += err
    seq_err += sq_err
  errors = float(low_batch) * float(errors) / batch_size
  seq_err = float(low_batch) * float(seq_err) / batch_size
  data.print_out("  %s len %d errors %.2f sequence-errors %.2f"
                 % (task, l, 100*errors, 100*seq_err))
  return errors, seq_err, result

class Timer(object):
  """Utility class for tracking time used in a function"""
  def __init__(self, label, print_fn=data.print_out):
    self.startt = time.time()
    self.label = label
    self.print_fn = print_fn
    self.print_fn('Start %s' % self.label)

  def done(self):
    self.print_fn('Finish %s, took %s seconds' % (self.label, time.time()-self.startt))

def train_for_a_bit(sess, model, batch_size, nsteps, thresh=0.0):
  results_record = records.ResultsRecord(batch_size)
  for _ in range(nsteps):

    batch, within_bounds = model.curriculum.draw_example(batch_size)

    # Run a step and time it.
    start_time = time.time()
    result = model.step(sess, batch, True)

    # Accumulate statistics only if we did not exceed curriculum length.
    results_record.feed(result, time.time() - start_time, within_bounds)

  global_step, lr, pull = sess.run( [model.global_step, model.lr, model.pull])
  # Normalize and print out accumulated statistics.
  message = ('step %s ' % (global_step, ) +
             'len %s ' % model.curriculum.length_str +
             'lr %.8f pull %.3f ' % (lr, pull) +
             '%s' % str(results_record)
  )
  data.print_out(message)
  print(message, file=step_output)
  if FLAGS.do_batchnorm:
    mytf.print_bn_state(sess, model.config.nmaps)

  would_extend = model.curriculum.consider_extending(results_record)
  decent = (would_extend >= 1)
  extended = (would_extend >= 2)
  # If errors are below the curriculum threshold, move curriculum forward.
  if decent:
    # Either increase pull or, if it's large, average parameters.
    if pull < 0.1:
      sess.run(model.pull_incr_op)
    else:
      data.print_out("  Averaging parameters.")
      sess.run(model.avg_op)

  # Lower learning rate if we're worse than the last 3 checkpoints.
  # [XXX the logic isn't great in mixed-task settings; it picks one
  # task semi-arbitrary.]
  first_record = sorted(results_record.record_for_task.items())[0][1]
  acc_perp = data.safe_exp(first_record.avg_loss)
  if acc_perp > thresh:
    data.print_out("Lower learning rate: %s %s" % (acc_perp, thresh))
    sess.run(model.lr_decay_op)
  return (extended, acc_perp)

def run_evaluation(sess, model, batch_size):
  global_step, = sess.run( [model.global_step])
  for task in model.curriculum.tasks():
    errors = []
    for batch, length in model.curriculum.test_examples(batch_size, task):
      _, seq_err, result = single_test(length, model, sess, task,
                                       FLAGS.nprint, batch_size, batch=batch)
      errors.append(seq_err)
      if len(errors) >= 4 and min(errors[-4:]) == 1:
        break
    if FLAGS.print_one:
      data.print_out(result.to_string(0))
    if seq_err < 0.05 or FLAGS.always_large:  # Run larger test if we're good enough.
      _, seq_err, result = multi_test(data.forward_max, model, sess, task,
                              FLAGS.nprint, batch_size * 4)
      data.print_out("LARGE ERROR: %s %s %s"  % (global_step, seq_err, task))
      log_output.write('%s %s %s\n' % (global_step, seq_err, task))
      if FLAGS.print_one:
        data.print_out(result.to_string(0))
  if seq_err < 0.01:  # Super-large test on 1-task large-forward models.
    if data.forward_max > 4000 and len(tasks) == 1:
      multi_test(data.forward_max, model, sess, task, FLAGS.nprint,
                 batch_size * 16, 0)

def checkpoint(sess, model, checkpoint_dir):
  checkpoint_path = os.path.join(checkpoint_dir, "neural_gpu.ckpt")
  global_step, = sess.run( [model.global_step])
  model.saver.save(sess, checkpoint_path,
                   global_step=model.global_step,
                   write_meta_graph=False)
  with open(os.path.join(checkpoint_dir, 'neural_gpu_curriculum.ckpt'), 'w') as f:
    yaml.dump(model.curriculum, f)


def train_loop(sess, model, batch_size, checkpoint_dir):
  time_till_ckpt = FLAGS.time_till_ckpt
  time_till_eval = FLAGS.time_till_eval
  # Main training loop.
  accuracies = [1e4]*3
  while True:
    data.print_out("Reminder: checkpoint dir %s" % checkpoint_dir)
    timer = Timer("training steps")
    extended, acc = train_for_a_bit(sess, model, batch_size, FLAGS.steps_per_epoch,
                                    max(accuracies[-3:]))
    accuracies.append(acc)
    if extended: # If we extended, don't just lower the learning rate
      accuracies.append(1000) 
    timer.done()

    # Save checkpoint.
    time_till_ckpt -= 1
    if time_till_ckpt == 0:
      time_till_ckpt = FLAGS.time_till_ckpt
      timer = Timer("saving checkpoint")
      checkpoint(sess, model, checkpoint_dir)
      timer.done()

    # Run evaluation.
    global_step, = sess.run( [model.global_step])
    time_till_eval -= 1
    if time_till_eval == 0:
      time_till_eval = FLAGS.time_till_eval
      timer = Timer("running evaluation %s"  % global_step)
      run_evaluation(sess, model, batch_size)
      timer.done()

    global_step, = sess.run( [model.global_step])
    if FLAGS.max_steps and global_step  >= FLAGS.max_steps:
      data.print_out("Finished all %s steps" % global_step)
      checkpoint(sess, model, checkpoint_dir)
      break

def start_and_train():
  """Train the model."""
  with tf.Session() as sess:
    timer = Timer('initialization')
    model = initialize(sess)
    timer.done()
    train_loop(sess, model, FLAGS.batch_size, get_checkpoint_dir())
