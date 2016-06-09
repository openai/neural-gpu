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
"""Convolutional Gated Recurrent Networks for Algorithm Learning."""

import math
import random
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

bins = [8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64, 128]
all_tasks = ["sort", "kvsort", "id", "rev", "rev2", "incr", "add", "left",
             "right", "left-shift", "right-shift", "bmul", "mul", "dup",
             "badd", "qadd", "search"]
forward_max = 128
log_filename = ""


def pad(l):
  for b in bins:
    if b >= l: return b
  return forward_max


train_set = {}
test_set = {}
for some_task in all_tasks:
  train_set[some_task] = []
  test_set[some_task] = []
  for all_max_len in xrange(10000):
    train_set[some_task].append([])
    test_set[some_task].append([])


def add(n1, n2, base=10):
  """Add two numbers represented as lower-endian digit lists."""
  k = max(len(n1), len(n2)) + 1
  d1 = n1 + [0 for _ in xrange(k - len(n1))]
  d2 = n2 + [0 for _ in xrange(k - len(n2))]
  res = []
  carry = 0
  for i in xrange(k):
    if d1[i] + d2[i] + carry < base:
      res.append(d1[i] + d2[i] + carry)
      carry = 0
    else:
      res.append(d1[i] + d2[i] + carry - base)
      carry = 1
  while res and res[-1] == 0:
    res = res[:-1]
  if res: return res
  return [0]

def to_base(num, b):
  ans = []
  while num:
    ans.append(num%b)
    num //= b
  return list(ans or [0])

def from_base(lst, b):
  num = 0
  for v in lst[::-1]:
    num = num*b + v
  return num

generators = {}

class DataGenerator(object):
  nclass = 33

  def rand_pair(self, length):
    """Random data pair for a task. Total length should be <= l."""
    raise NotImplementedError()

  def rand_triple(self, length):
    inpt, outp = self.rand_pair(length)
    return inpt, outp, 0

  def get_batch(self, length, batch_size):
    pad_length = pad(length)
    data_triples = [self.rand_triple(length) for _ in xrange(batch_size)]
    padded_data = np.array([[x + [0]*(pad_length - len(x))
                             for x in lst[:2]]
                            for lst in data_triples])
    # batch_size x 3 x length
    # -> 3 x length x batch_size
    inpt, outp = padded_data.transpose([1,2,0])
    return inpt, outp, [x[2] for x in data_triples]

  def _initialize(self, nclass):
    self.nclass = nclass


class OpGenerator(DataGenerator):
  def __init__(self, base, f, sep):
    self.base = base
    self.f = f
    self.sep = sep

  def rand_pair(self, l):
    k = (l-1)//2
    n1 = random.randint(0, self.base**k-1)
    n2 = random.randint(0, self.base**k-1)
    result = self.f(n1, n2)
    inp = ([x+1 for x in to_base(n1, self.base)] + [self.sep] +
           [x+1 for x in to_base(n2, self.base)])
    outp = [x+1 for x in to_base(result, self.base)]
    return inp, outp

generators.update(dict(add=OpGenerator(10, np.add, 11),
                       badd=OpGenerator(2, np.add, 11),
                       qadd=OpGenerator(4, np.add, 11),
                       mul=OpGenerator(10, np.multiply, 12),
                       bmul=OpGenerator(2, np.multiply, 12),
                       qmul=OpGenerator(4, np.multiply, 12)))

class FGenerator(DataGenerator):
  def __init__(self, f):
    self.f = f

  def rand_pair(self, l):
    x = np.random.randint(self.nclass - 1, size=l) + 1
    return list(x), list(self.f(x))

generators.update(dict(rev=FGenerator(lambda l: l[::-1]),
                       sort=FGenerator(sorted),
                       id=FGenerator(lambda l: l),
                       ))


class DupGenerator(DataGenerator):
  def rand_pair(self, l):
    k = l/2
    x = [np.random.randint(self.nclass - 1) + 1 for _ in xrange(k)]
    inp = x + [0 for _ in xrange(l - k)]
    res = x + x + [0 for _ in xrange(l - 2*k)]
    return inp, res

class MixGenerator(DataGenerator):
  def __init__(self, gens):
    self.sets = gens

  def rand_triple(self, length):
    i = np.random.randint(len(self.sets))
    pair = self.sets[i].rand_pair(length)
    return pair[0], pair[1], i

generators.update(dict(dup=DupGenerator(),
                       mix=MixGenerator([generators[x] for x in 'badd bmul'.split()]),
                       ))

def init_data(task, length, nbr_cases, nclass):
  """Data initialization."""
  assert False
  print "Creating cases", nbr_cases, task, length, nclass
  def rand_rev2_pair(l):
    """Random data pair for reverse2 task. Total length should be <= l."""
    inp = [(np.random.randint(nclass - 1) + 1,
            np.random.randint(nclass - 1) + 1) for _ in xrange(l/2)]
    res = [i for i in reversed(inp)]
    return [x for p in inp for x in p], [x for p in res for x in p]

  def rand_search_pair(l):
    """Random data pair for search task. Total length should be <= l."""
    inp = [(np.random.randint(nclass - 1) + 1,
            np.random.randint(nclass - 1) + 1) for _ in xrange(l-1/2)]
    q = np.random.randint(nclass - 1) + 1
    res = 0
    for (k, v) in reversed(inp):
      if k == q:
        res = v
    return [x for p in inp for x in p] + [q], [res]

  def rand_kvsort_pair(l):
    """Random data pair for key-value sort. Total length should be <= l."""
    keys = [(np.random.randint(nclass - 1) + 1, i) for i in xrange(l/2)]
    vals = [np.random.randint(nclass - 1) + 1 for _ in xrange(l/2)]
    kv = [(k, vals[i]) for (k, i) in keys]
    sorted_kv = [(k, vals[i]) for (k, i) in sorted(keys)]
    return [x for p in kv for x in p], [x for p in sorted_kv for x in p]

  def spec(inp):
    """Return the target given the input for some tasks."""
    if False:
      pass
    elif task == "id":
      return inp
    elif task == "rev":
      return [i for i in reversed(inp)]
    elif task == "incr":
      carry = 1
      res = []
      for i in xrange(len(inp)):
        if inp[i] + carry < nclass:
          res.append(inp[i] + carry)
          carry = 0
        else:
          res.append(1)
          carry = 1
      return res
    elif task == "left":
      return [inp[0]]
    elif task == "right":
      return [inp[-1]]
    elif task == "left-shift":
      return [inp[l-1] for l in xrange(len(inp))]
    elif task == "right-shift":
      return [inp[l+1] for l in xrange(len(inp))]
    else:
      print_out("Unknown spec for task " + str(task))
      sys.exit()

  l = length
  cur_time = time.time()
  total_time = 0.0
  for case in xrange(nbr_cases):
    total_time += time.time() - cur_time
    cur_time = time.time()
    if l > 10000 and case % 100 == 1:
      print_out("  avg gen time %.4f s" % (total_time / float(case)))
    if task in ["add", "badd", "qadd", "bmul", "mul"]:
      i, t = rand_pair(l, task)
      train_set[task][len(i)].append([i, t])
      i, t = rand_pair(l, task)
      test_set[task][len(i)].append([i, t])
    elif task == "dup":
      i, t = rand_dup_pair(l)
      train_set[task][len(i)].append([i, t])
      i, t = rand_dup_pair(l)
      test_set[task][len(i)].append([i, t])
    elif task == "rev2":
      i, t = rand_rev2_pair(l)
      train_set[task][len(i)].append([i, t])
      i, t = rand_rev2_pair(l)
      test_set[task][len(i)].append([i, t])
    elif task == "search":
      i, t = rand_search_pair(l)
      train_set[task][len(i)].append([i, t])
      i, t = rand_search_pair(l)
      test_set[task][len(i)].append([i, t])
    elif task == "kvsort":
      i, t = rand_kvsort_pair(l)
      train_set[task][len(i)].append([i, t])
      i, t = rand_kvsort_pair(l)
      test_set[task][len(i)].append([i, t])
    else:
      inp = [np.random.randint(nclass - 1) + 1 for i in xrange(l)]
      target = spec(inp)
      train_set[task][l].append([inp, target])
      inp = [np.random.randint(nclass - 1) + 1 for i in xrange(l)]
      target = spec(inp)
      test_set[task][l].append([inp, target])


def to_symbol(i):
  """Covert ids to text."""
  if i == 0: return ""
  if i == 11: return "+"
  if i == 12: return "*"
  return str(i-1)


def to_id(s):
  """Covert text to ids."""
  if s == "+": return 11
  if s == "*": return 12
  return int(s) + 1


def get_batch(max_length, batch_size, do_train, task, offset=None, preset=None):
  return generators[task].get_batch(max_length, batch_size)

def get_batch_old(max_length, batch_size, do_train, task, offset=None, preset=None):
  """Get a batch of data, training or testing."""
  inputs = []
  targets = []
  length = max_length
  if preset is None:
    cur_set = test_set[task]
    if do_train: cur_set = train_set[task]
    while not cur_set[length]:
      length -= 1
  pad_length = pad(length)
  for b in xrange(batch_size):
    if preset is None:
      elem = random.choice(cur_set[length])
      if offset is not None and offset + b < len(cur_set[length]):
        elem = cur_set[length][offset + b]
    else:
      elem = preset
    inp, target = elem[0], elem[1]
    assert len(inp) == length
    inputs.append(inp + [0 for l in xrange(pad_length - len(inp))])
    targets.append(target + [0 for l in xrange(pad_length - len(target))])
  res_input = []
  res_target = []
  for l in xrange(pad_length):
    new_input = np.array([inputs[b][l] for b in xrange(batch_size)],
                         dtype=np.int32)
    new_target = np.array([targets[b][l] for b in xrange(batch_size)],
                          dtype=np.int32)
    res_input.append(new_input)
    res_target.append(new_target)
  return res_input, res_target


def print_out(s, newline=True):
  """Print a message out and log it to file."""
  if log_filename:
    try:
      with gfile.GFile(log_filename, mode="a") as f:
        f.write(s + ("\n" if newline else ""))
    # pylint: disable=bare-except
    except:
      sys.stdout.write("Error appending to %s\n" % log_filename)
  sys.stdout.write(s + ("\n" if newline else ""))
  sys.stdout.flush()


def decode(output):
  return [np.argmax(o, axis=1) for o in output]


def accuracy(inpt, output, target, batch_size, nprint):
  """Calculate output accuracy given target."""
  assert nprint < batch_size + 1
  def task_print(inp, output, target):
    stop_bound = 0
    print_len = 0
    while print_len < len(target) and target[print_len] > stop_bound:
      print_len += 1
    print_out("    i: " + " ".join([str(i - 1) for i in inp if i > 0]))
    print_out("    o: " +
              " ".join([str(output[l] - 1) for l in xrange(print_len)]))
    print_out("    t: " +
              " ".join([str(target[l] - 1) for l in xrange(print_len)]))
  decoded_target = target
  decoded_output = decode(output)
  total = 0
  errors = 0
  seq = [0 for b in xrange(batch_size)]
  for l in xrange(len(decoded_output)):
    for b in xrange(batch_size):
      if decoded_target[l][b] > 0:
        total += 1
        if decoded_output[l][b] != decoded_target[l][b]:
          seq[b] = 1
          errors += 1
  e = 0  # Previous error index
  for _ in xrange(min(nprint, sum(seq))):
    while seq[e] == 0:
      e += 1
    task_print([inpt[l][e] for l in xrange(len(inpt))],
               [decoded_output[l][e] for l in xrange(len(decoded_target))],
               [decoded_target[l][e] for l in xrange(len(decoded_target))])
    e += 1
  for b in xrange(nprint - errors):
    task_print([inpt[l][b] for l in xrange(len(inpt))],
               [decoded_output[l][b] for l in xrange(len(decoded_target))],
               [decoded_target[l][b] for l in xrange(len(decoded_target))])
  return errors, total, sum(seq)


def safe_exp(x):
  perp = 10000
  if x < 100: perp = math.exp(x)
  if perp > 10000: return 10000
  return perp

def reversible_flatten(lst):
  lengths = []
  result = []
  for w in lst:
    if isinstance(w, list):
      lengths.append(len(w))
      result.extend(w)
    else:
      lengths.append(-1)
      result.append(w)
  return result, lengths

def reverse_flatten(flattened, lengths):
  ans = []
  i = 0
  for l in lengths:
    if l == -1:
      ans.append(flattened[i])
      i += 1
    else:
      ans.append(flattened[i:i+l])
      i += l
  return ans

def sess_run_dict(sess, fetches, feed_dict):
  keys = list(fetches.keys())
  wanted_input_list = [fetches[k] for k in keys]
  input_list, lengths = reversible_flatten(wanted_input_list)
  res = sess.run(input_list, feed_dict)
  unflattened_res = reverse_flatten(res, lengths)
  return dict(zip(keys, unflattened_res))
