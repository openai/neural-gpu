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

PADDING = False


def pad(l):
  for b in bins + [forward_max]:
    if b >= l: return b
  raise IndexError("Length %s longer than max length %s" % (l, forward_max))

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
    return [np.concatenate([x, np.zeros((pad_length - x.shape[0],) + x.shape[1:])],
                           axis=0)
            for x in data]

  def get_batch(self, length, batch_size):
    inps, outps = [], []
    for _ in xrange(batch_size):
      inp, outp = self.rand_pair_padded(length)
      inps.append(inp)
      outps.append(outp)

    inp = np.stack(inps, 0)
    outp = np.stack(outps, 0)
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
    return l%2 == 1 and l > 3

  def _rand_inputs(self, k):
    k = int(k)
    n1 = random.randint(0, self.base**k-1)
    n2 = random.randint(0, self.base**k-1)
    return (n1, n2)

  def rand_pair(self, l):
    k = int((l-1 - 2*PADDING)//2)
    n1, n2 = self._rand_inputs(k)
    result = self.f(n1, n2)
    inp = np.concatenate([[21] if PADDING else [],
       to_base(n1, self.base, k) + 1,
       [self.sep],
                          to_base(n2, self.base, k) + 1, [22] if PADDING else []])
    outp = np.concatenate([[21] if PADDING else [],
            to_base(result, self.base, 2*k+1) + 1,
                           [22] if PADDING else []])
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


class AlignedOpGenerator(OpGenerator):
  def rand_pair(self, l):
    k = int((l-1 - 2*PADDING)//2)
    n1, n2 = self._rand_inputs(k)
    result = self.f(n1, n2)
    n1, n2 = [np.concatenate([[21] if PADDING else [],
                              to_base(n, self.base, k) + 1,
                              [22] if PADDING else []]) for n in [n1,n2]]
    preferred_length = l#max(len(n1), len(n2))+1
    pad_n1, pad_n2 = [np.pad(n,(0, preferred_length-len(n)), 'constant') for n in (n1, n2)]
    pad_n2[len(n2)] = self.sep
    inp2 = np.vstack([pad_n1, pad_n2])
    #XXX cheating on length here
    if False:
      o = np.concatenate([[21] if PADDING else [], to_base(result, self.base, l-1) + 1])
    else:
      #o = to_base(result, self.base) + 1
      o = to_base(result, self.base, k+2) + 1
    outp = np.pad(o, (0, preferred_length - len(o)), 'constant')
    return inp2.transpose(), outp

class AlignedToughAddGenerator(AlignedOpGenerator, ToughAddGenerator):
  pass

generators.update(dict(badde=AlignedOpGenerator(2, operator.add, 11),
                       qadde=AlignedOpGenerator(4, operator.add, 12),
                       adde=AlignedOpGenerator(10, operator.add, 13),
                       baddet=AlignedToughAddGenerator(2, 11),
                       qaddet=AlignedToughAddGenerator(4, 12),
                       addet=AlignedToughAddGenerator(10, 13),
))

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

@np.vectorize
def char_to_symbol(i):
  """Covert ids to text."""
  if i == 0: return " "
  if i in [11,12,13]: return "+"
  if i in [14,15,16]: return "*"
  return str(i-1)

def join_array(array):
  if len(array.shape) == 1:
    return ''.join(array).rstrip(' ')
  elif len(array.shape) == 2:
    return '\n'.join([''.join(a).rstrip(' ') for a in array])
  else:
    raise ValueError("Weird shape for joining: %s" % array.shape)

def to_string(array):
  return join_array(char_to_symbol(array))

@np.vectorize
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

def load_class(name):
  modulename, classname = name.rsplit('.', 1)
  module = __import__(modulename)
  return getattr(module, classname)
