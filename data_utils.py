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
import operator

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

def to_base(num, b, l=1):
  assert num >= 0
  ans = []
  while num:
    ans.append(num%b)
    num //= b
  while len(ans) < l:
    ans.append(0)
  return np.array(ans)

def from_base(lst, b):
  num = 0
  for v in lst[::-1]:
    num = num*b + v
  return num

generators = {}

class DataGenerator(object):
  nclass = 33
  name = '<unknown task>'
  taskid = 0

  def is_valid_length(self, l):
    return True

  def rand_pair(self, length):
    """Random data pair for a task. Total length should be <= l."""
    raise NotImplementedError()

  def rand_pair_padded(self, length):
    pad_length = pad(length)
    data = self.rand_pair(length)
    return [np.concatenate([x, np.zeros(pad_length - len(x))], axis=-1)
            for x in data]

  def get_batch(self, length, batch_size):
    result = np.array([self.rand_pair_padded(length)
                       for _ in xrange(batch_size)])
    inp, outp = result.transpose([1,0,2])
    return inp, outp, np.array([self.taskid] * batch_size)

  def _initialize(self, nclass):
    self.nclass = nclass

  def __repr__(self):
    return "<%s name='%s' taskid=%s>" % (self.__class__.__name__, self.name, self.taskid)

class OpGenerator(DataGenerator):
  def __init__(self, base, f, sep):
    self.base = base
    self.f = f
    self.sep = sep

  def is_valid_length(self, l):
    return l%2 == 1 and l > 1

  def _rand_inputs(self, k):
    n1 = random.randint(0, self.base**k-1)
    n2 = random.randint(0, self.base**k-1)
    return (n1, n2)

  def rand_pair(self, l):
    k = (l-1)//2
    n1, n2 = self._rand_inputs(k)
    result = self.f(n1, n2)
    inp = np.concatenate([to_base(n1, self.base, k) + 1,
                          [self.sep],
                          to_base(n2, self.base, k) + 1])
    outp = to_base(result, self.base, 2*k+1) + 1
    return inp, outp

generators.update(dict(badd=OpGenerator(2, operator.add, 11),
                       qadd=OpGenerator(4, operator.add, 12),
                       add=OpGenerator(10, operator.add, 13),
                       bmul=OpGenerator(2, operator.mul, 14),
                       qmul=OpGenerator(4, operator.mul, 15),
                       mul=OpGenerator(10, operator.mul, 16),))


class ToughAddGenerator(OpGenerator):
  def __init__(self, base, sep):
    super(ToughAddGenerator, self).__init__(base, operator.add, sep)

  def _rand_inputs(self, k):
    r = random.random()
    if r < 0.2:
      lo, hi = sorted([random.randint(1, k), random.randint(1, k)])
      vals = (self.base**hi - self.base**(lo-1), random.randint(0,self.base**(lo)-1))
    elif r < .4:
      k2 = random.choice([k, random.randint(1, k)])
      lo = random.randint(1, self.base**k2-1)
      vals = (lo, self.base**k2 - lo - random.randint(0,1))
    else:
      vals = (random.randint(0, self.base**k-1), random.randint(0, self.base**k-1))
    if random.random() > .5:
      return vals
    else:
      return vals[::-1]

generators.update(dict(baddt=ToughAddGenerator(2, 11),
                       qaddt=ToughAddGenerator(4, 11),))

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

  def rand_pair(self, length):
    i = np.random.randint(len(self.sets))
    return self.sets[i].rand_pair(length)

generators.update(dict(dup=DupGenerator(),
                       mix=MixGenerator([generators[x] for x in 'badd bmul'.split()]),
                       ))

for k in generators:
  generators[k].name = k

def to_symbol(i):
  """Covert ids to text."""
  if i == 0: return ""
  if i in [11,12,13]: return "+"
  if i in [14,15,16]: return "*"
  return str(i-1)

def to_id(s):
  """Covert text to ids."""
  if s == "+": return 11
  if s == "*": return 14
  return int(s) + 1


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

def safe_exp(x):
  perp = 10000
  if x < 100: perp = math.exp(x)
  if perp > 10000: return 10000
  return perp
