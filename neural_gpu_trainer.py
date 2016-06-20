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

import data_utils as data
import neural_gpu
import neural_curriculum
import mytf

def define_flags():
  tf.app.flags.DEFINE_float("lr", 0.001, "Learning rate.")
  tf.app.flags.DEFINE_float("init_weight", 1.0, "Initial weights deviation.")
  tf.app.flags.DEFINE_float("max_grad_norm", 1.0, "Clip gradients to this norm.")
  tf.app.flags.DEFINE_float("cutoff", 1.2, "Cutoff at the gates.")
  tf.app.flags.DEFINE_float("cutoff_tanh", 0.0, "Cutoff at tanh.")
  tf.app.flags.DEFINE_float("pull", 0.0005, "Starting pull of the relaxations.")
  tf.app.flags.DEFINE_float("pull_incr", 1.2, "Increase pull by that much.")
  tf.app.flags.DEFINE_float("curriculum_bound", 0.15, "Move curriculum < this.")
  tf.app.flags.DEFINE_float("dropout", 0.15, "Dropout that much.")
  tf.app.flags.DEFINE_float("grad_noise_scale", 0.0, "Gradient noise scale.")
  tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size.")
  tf.app.flags.DEFINE_integer("low_batch_size", 16, "Low batch size.")
  tf.app.flags.DEFINE_integer("steps_per_epoch", 200, "Steps per epoch.")
  tf.app.flags.DEFINE_integer("nmaps", 24, "Number of floats in each cell.")
  tf.app.flags.DEFINE_integer("niclass", 33, "Number of classes (0 is padding).")
  tf.app.flags.DEFINE_integer("noclass", 33, "Number of classes (0 is padding).")
  tf.app.flags.DEFINE_integer("train_data_size", 5000, "Training examples/len.")
  tf.app.flags.DEFINE_integer("max_length", 21, "Maximum length.")
  tf.app.flags.DEFINE_integer("rx_step", 6, "Relax that many recursive steps.")
  tf.app.flags.DEFINE_integer("random_seed", 125459, "Random seed.")
  tf.app.flags.DEFINE_integer("time_till_ckpt", 30, "How many tests per checkpoint")
  tf.app.flags.DEFINE_integer("nconvs", 2, "How many convolutions / 1 step.")
  tf.app.flags.DEFINE_integer("kw", 3, "Kernel width.")
  tf.app.flags.DEFINE_integer("kh", 3, "Kernel height.")
  tf.app.flags.DEFINE_integer("height", 4, "Height.")
  tf.app.flags.DEFINE_integer("forward_max", 101, "Maximum forward length.")
  tf.app.flags.DEFINE_integer("jobid", 0, "Task id when running on borg.")
  tf.app.flags.DEFINE_integer("nprint", 0, "How many test examples to print out.")
  tf.app.flags.DEFINE_integer("mode", 0, "Mode: 0-train other-decode.")
  tf.app.flags.DEFINE_bool("animate", False, "Whether to produce an animation.")
  tf.app.flags.DEFINE_float("smooth_grad", 0.0, "Whether to avoid clipping gradient")
  tf.app.flags.DEFINE_float("smooth_grad_tanh", 0.0, "Whether to avoid clipping tanh gradient")
  tf.app.flags.DEFINE_string("task", "rev", "Which task are we learning?")
  tf.app.flags.DEFINE_string("train_dir", "/tmp/neural", "Directory to store models.")

  #tf.app.flags.DEFINE_bool("do_attention", False, "Whether to use attention method.")
  tf.app.flags.DEFINE_integer("num_attention", 0, "Number of attention modules to use.")

  tf.app.flags.DEFINE_integer("do_batchnorm", 0, "Whether to use batch normalization.")
  tf.app.flags.DEFINE_bool("do_resnet", False, "Whether to use resnets.")

  tf.app.flags.DEFINE_bool("do_lastout", False, "Whether to use last output.")
  tf.app.flags.DEFINE_bool("do_layers", False, "Expose output for all layers.")

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
      raise ValueError("Bad log dir")
    else:
      print()
      raise ValueError("Even though the argv didn't change, we'll still kill you.")

  with open(command_fname, 'w') as f:
    f.write(' '.join(sys.argv)+'\n')

  with open(os.path.join(checkpoint_dir, 'all_args'), 'w') as f:
    yaml.dump(FLAGS.__flags, f, default_flow_style=False)

  with open(os.path.join(checkpoint_dir, 'git-rev'), 'w') as f:
    subprocess.call(['git', 'rev-parse', 'HEAD'], stdout=f)

  log_output = open(os.path.join(checkpoint_dir, 'results'), 'w', 1)
  step_output = open(os.path.join(checkpoint_dir, 'steps'), 'w', 1)

def load_model(sess, checkpoint_dir):
  # possibly tf.reset_default_graph()
  with open(os.path.join(checkpoint_dir, 'all_args')) as f:
    options = yaml.load(f)
  FLAGS._parse_flags()
  FLAGS.__flags.update(options)
  data.forward_max = max(FLAGS.forward_max, data.bins[-1])
  config = neural_curriculum.NeuralConfig(FLAGS)
  model = neural_gpu.NeuralGPU(config)
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    model.saver.restore(sess, ckpt.model_checkpoint_path)
  return model

def get_checkpoint_dir():
  return FLAGS.train_dir + ('-seed%s-pid%s' % (FLAGS.random_seed, os.getpid()))

def get_config_from_flags(checkpoint_dir = None):
  # Set random seed.
  seed = FLAGS.random_seed + max(0, FLAGS.jobid)
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

  if FLAGS.jobid >= 0:
    data.log_filename = os.path.join(checkpoint_dir, "log%d" % FLAGS.jobid)

  data.print_out("NN ", newline=False)

  config = neural_curriculum.NeuralConfig(FLAGS)

  # Check data sizes.
  while len(data.bins) > 1 and data.bins[-2] > config.max_length + EXTRA_EVAL:
    data.bins = data.bins[:-1]
  assert data.bins[0] > FLAGS.rx_step
  data.forward_max = max(FLAGS.forward_max, data.bins[-1])

  return config

def initialize(sess, checkpoint_dir=None):
  """Initialize data and model."""
  config = get_config_from_flags(checkpoint_dir)
  data.print_out(str(config))

  if checkpoint_dir is None:
    checkpoint_dir = get_checkpoint_dir()
  log_parameters(checkpoint_dir)

  # Initialize data for each task.
  nclass = min(config.niclass, config.noclass)
  tasks = config.task.split(",")
  data_generators = [data.generators[t] for t in tasks]
  for g in data_generators:
    g._initialize(nclass)
  #data_size = FLAGS.train_data_size if FLAGS.mode == 0 else 1000
  #goal_lengths = [l for l in xrange(max_length + EXTRA_EVAL - 1)] + [data.forward_max]
  # for t in tasks:
  #   for l in xrange(max_length + EXTRA_EVAL - 1):
  #     data.init_data(t, l, data_size, nclass)
  #   data.init_data(t, data.bins[-2], data_size, nclass)
  #   data.init_data(t, data.bins[-1], data_size, nclass)
  #   end_size = 4 * 1024 if FLAGS.mode > 0 else 1024
  #   data.init_data(t, data.forward_max, end_size, nclass)

  # Create model and initialize it.
  tf.get_variable_scope().set_initializer(
      tf.uniform_unit_scaling_initializer(factor=1.8 * FLAGS.init_weight))
  model = neural_gpu.NeuralGPU(config)
  data.print_out("Created model.")
  sess.run(tf.initialize_all_variables())
  model.renormalize(sess)
  data.print_out("Initialized variables.")

  # Load model from parameters if a checkpoint exists.
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    data.print_out("Reading model parameters from %s"
                   % ckpt.model_checkpoint_path)
    model.saver.restore(sess, ckpt.model_checkpoint_path)

    #curriculum = neural_curriculum.MixedCurriculum(data_generators, model.config)
  curriculum = neural_curriculum.GeneralizeCurriculum(data_generators, model.config)
  model.curriculum = curriculum

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
  for mstep in xrange(batch_size / low_batch):
    cur_offset = None if offset is None else offset + mstep * low_batch
    err, sq_err, _ = single_test(l, model, sess, task, to_print, low_batch,
                                 False, cur_offset)
    to_print = max(0, to_print - low_batch)
    errors += err
    seq_err += sq_err
    if FLAGS.mode > 0:
      cur_errors = float(low_batch * errors) / ((mstep+1) * low_batch)
      cur_seq_err = float(low_batch * seq_err) / ((mstep+1) * low_batch)
      data.print_out("    %s multitest current errors %.2f sequence-errors %.2f"
                     % (task, 100*cur_errors, 100*cur_seq_err))
  errors = float(low_batch) * float(errors) / batch_size
  seq_err = float(low_batch) * float(seq_err) / batch_size
  data.print_out("  %s len %d errors %.2f sequence-errors %.2f"
                 % (task, l, 100*errors, 100*seq_err))
  return errors, seq_err

class Timer(object):
  def __init__(self, label, print_fn=data.print_out):
    self.startt = time.time()
    self.label = label
    self.print_fn = print_fn
    self.print_fn('Start %s' % self.label)

  def done(self):
    self.print_fn('Finish %s, took %s seconds' % (self.label, time.time()-self.startt))

def train_for_a_bit(sess, model, batch_size, nsteps, thresh=0.0):
  curriculum = model.curriculum
  global_step, = sess.run( [model.global_step])
  results_record = neural_curriculum.ResultsRecord(batch_size)
  for _ in xrange(nsteps):
    global_step += 1

    batch, within_bounds = model.curriculum.draw_example(batch_size)

    # Run a step and time it.
    start_time = time.time()
    result = model.step(sess, batch, True)

    # Accumulate statistics only if we did not exceed curriculum length.
    results_record.feed(result, time.time() - start_time, within_bounds)

  # Normalize and print out accumulated statistics.
  message = ('step %s ' % (global_step, ) +
             'len %s ' % curriculum.length_str +
             'lr %.8f pull %.3f ' % (model.lr, model.pull) +
             '%s' % str(results_record)
  )
  data.print_out(message)
  print(message, file=step_output)
  if FLAGS.do_batchnorm:
    mytf.print_bn_state(sess, model.config.nmaps)

  would_extend = curriculum.consider_extending(results_record)
  decent = (would_extend >= 1)
  extended = (would_extend >= 2)
  # If errors are below the curriculum threshold, move curriculum forward.
  if decent:
    # Either increase pull or, if it's large, average parameters.
    if model.pull < 0.1:
      model.pull *= model.config.pull_incr
    else:
      data.print_out("  Averaging parameters.")
      sess.run(model.avg_op)
      # XXX this used to exist, but it doesn't really make sense
      #if results_record.values()[0].avg_seq_err < (model.config.curriculum_bound / 3.0):
      #  model.lr *= 0.98

  # Lower learning rate if we're worse than the last 3 checkpoints.
  # XXX improve this in a mixed setting
  acc_perp = data.safe_exp(results_record.record_for_task[0].avg_loss)
  if acc_perp > thresh:
    data.print_out("Lower learning rate: %s %s" % (acc_perp, thresh))
    model.lr *= 0.98
  return (extended, acc_perp)

def run_evaluation(sess, model, batch_size):
  global_step, = sess.run( [model.global_step])
  for task in model.curriculum.tasks():
    errors = []
    for batch, length in model.curriculum.test_examples(batch_size, task):
      _, seq_err, _ = single_test(length, model, sess, task,
                                  FLAGS.nprint, batch_size, batch=batch)
      errors.append(seq_err)
      if len(errors) >= 4 and min(errors[-4:]) == 1:
        break
    if seq_err < 0.05:  # Run larger test if we're good enough.
      _, seq_err = multi_test(data.forward_max, model, sess, task,
                              FLAGS.nprint, batch_size * 4)
      data.print_out("LARGE ERROR: %s %s %s"  % (global_step, seq_err, task))
      log_output.write('%s %s %s\n' % (global_step, seq_err, task))
  if seq_err < 0.01:  # Super-large test on 1-task large-forward models.
    if data.forward_max > 4000 and len(tasks) == 1:
      multi_test(data.forward_max, model, sess, task, FLAGS.nprint,
                 batch_size * 16, 0)

def train_loop(sess, model, batch_size, checkpoint_dir):
  time_till_ckpt = FLAGS.time_till_ckpt
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
      checkpoint_path = os.path.join(checkpoint_dir, "neural_gpu.ckpt")
      global_step, = sess.run( [model.global_step])
      model.saver.save(sess, checkpoint_path,
                       global_step=model.global_step)
      timer.done()

    # Run evaluation.
    global_step, = sess.run( [model.global_step])
    timer = Timer("running evaluation %s"  % global_step)
    run_evaluation(sess, model, batch_size)
    timer.done()

def start_and_train():
  """Train the model."""
  with tf.Session() as sess:
    timer = Timer('initialization')
    model = initialize(sess)
    timer.done()
    train_loop(sess, model, FLAGS.batch_size, get_checkpoint_dir())


def animate(l, test_data, anim_size):
  """Create animation for the given data (hacky matplotlib use)."""
  xf = 12  # Extra frames to slow down at start and end.
  fps = 2  # Frames per step.

  # Make the figure.
  fig = plt.figure(figsize=(16, 9), facecolor="white")
  ax = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=2)
  ax.set_xticks([i * 24-0.5 for i in xrange(4)])
  ax.set_xticklabels([])
  ax.set_yticks([i - 0.5 for i in xrange(l+1)])
  ax.grid(which="major", axis="both", linestyle="-", color="black")
  # We need text fields.
  text_fields = []
  text_size = 24*32/l
  for y in xrange(l):
    text_fields.append(ax.text(
        11.25, y + 0.15, "", color="g", ha="center", va="center",
        bbox={"facecolor": "b", "alpha": 0.01, "pad": 24 * text_size},
        size=text_size - (4 * 32 / l), animated=True))
  im = ax.imshow(np.zeros_like(test_data[0][0][0]), vmin=-1.0,
                 vmax=1.0, cmap="gray", aspect="auto", origin="upper",
                 interpolation="none", animated=True)
  im.set_zorder(1)

  # Main animation step.
  def animation_update(frame_no, test_data, xf, im, text_fields):
    """Update an animation frame."""
    steps, inpt, out_raw = test_data
    length = len(steps)
    batch = frame_no / (fps * (l+4*xf))
    index = int((frame_no % (fps * (l+4*xf))) / fps)
    # Cut output after first padding.
    out = [out_raw[i][batch] for i in xrange(len(text_fields))]
    if 0 in out:
      i = out.index(0)
      out = out[0:i] + [0 for _ in xrange(len(out) - i)]
    # Show the state after the first frames.
    if index >= 2*xf:
      im.set_array(steps[min(length - 1, index - 2*xf)][batch])
      for i, t in enumerate(text_fields):
        if index - 2*xf < length:
          t.set_text("")
        else:
          t.set_text(data.to_symbol(out[i]))
    else:
      for i, t in enumerate(text_fields):
        t.set_text(data.to_symbol(inpt[i][batch]) if index < xf else "")
      if index < xf:
        im.set_array(np.zeros_like(steps[0][0]))
      else:
        im.set_array(steps[0][batch])
    return im,

  # Create the animation and save to mp4.
  animation = anim.FuncAnimation(
      fig, animation_update, blit=True, frames=(l+4*xf)*anim_size*fps,
      interval=500/fps, fargs=(test_data, xf, im, text_fields))
  animation.save("/tmp/neural_gpu.mp4", writer="mencoder", fps=4*fps, dpi=3*80)


def evaluate():
  """Evaluate an existing model."""
  batch_size = FLAGS.batch_size
  tasks = FLAGS.task.split(",")
  with tf.Session() as sess:
    model = initialize(sess)
    bound = data.bins[-1] + 1
    for t in tasks:
      l = model.config.min_length
      while l < model.config.max_length + EXTRA_EVAL and l < bound:
        _, seq_err, _ = single_test(l, model, sess, t, FLAGS.nprint,
                                    batch_size)
        l += 1
        while l < bound + 1 and not data.test_set[t][l]:
          l += 1
      # Animate.
      if FLAGS.animate:
        anim_size = 2
        _, _, result = single_test(l, model, sess, t, 0, anim_size,
                                      get_steps=True)
        this_is_broken_because_it_is_the_wrong_format
        animate(l, test_data, anim_size)
      # More tests.
      _, seq_err = multi_test(data.forward_max, model, sess, t, FLAGS.nprint,
                              batch_size * 4)
    if seq_err < 0.01:  # Super-test if we're very good and in large-test mode.
      if data.forward_max > 4000 and len(tasks) == 1:
        multi_test(data.forward_max, model, sess, tasks[0], FLAGS.nprint,
                   batch_size * 64, 0)

def main(_):
  if FLAGS.mode == 0:
    start_and_train()
  elif FLAGS.mode == 1:
    evaluate()
  elif FLAGS.mode == 2:
    with tf.Session() as sess:
      import cProfile as profile
      t = Timer("Starting...")
      profile.runctx('model = initialize(sess)', globals(), locals(), 'profile')
      t.done()
  else:
    interactive()

if __name__ == "__main__":
  tf.app.run()
