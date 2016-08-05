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
import functools
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


DIGITS = range(1, 11)
NULL = 0
PLUS = 11
MINUS = 12
TIMES = 13
DUP = 14
SPACE = 20
START = 21

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
  height = None
  min_length = 1

  def is_valid_length(self, l):
    return True

  def rand_pair(self, length):
    """Random data pair for a task. Total length should be <= l."""
    raise NotImplementedError()

  def rand_pair_padded(self, length):
    pad_length = pad(length)
    inp, outp = self.rand_pair(length)
    inp = np.array(inp)
    if len(inp.shape) == 1:
      inp = np.array([inp])
    padding_func = lambda x: np.pad(x, [(0,0)]*(len(x.shape)-1) +
                                       [(0, pad_length - x.shape[-1])], 'constant')
    inp, outp = padding_func(inp), padding_func(outp)
    assert inp.shape[-1] == pad_length, outp.shape[-1] == pad_length
    return inp, outp

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
  min_length = 3

  def __init__(self, base, f, sep, zero_pad=True):
    self.base = base
    self.f = f
    self.sep = sep
    self.zero_pad = zero_pad

  def is_valid_length(self, l):
    return l%2 == 1 and l >= self.min_length

  def _rand_inputs(self, k):
    k = int(k)
    n1 = random.randint(0, self.base**k-1)
    n2 = random.randint(0, self.base**k-1)
    return (n1, n2)

  def rand_pair(self, l):
    k = int((l-1 - 2*PADDING)//2)
    n1, n2 = self._rand_inputs(k)
    result = self.f(n1, n2)
    inp = np.concatenate([[START] if PADDING else [],
       to_base(n1, self.base, k if self.zero_pad else 1) + 1,
       [self.sep],
                          to_base(n2, self.base, k if self.zero_pad else 1) + 1,
                          #[22] if PADDING else []
    ])
    outp = np.concatenate([#[START] if PADDING else [],
            to_base(result, self.base, 2*k+1 if self.zero_pad else 1) + 1,
                           #[22] if PADDING else []
    ])
    return inp, outp

generators.update(dict(badd=OpGenerator(2, operator.add, 11),
                       qadd=OpGenerator(4, operator.add, 12),
                       add=OpGenerator(10, operator.add, 13),
                       bmul=OpGenerator(2, operator.mul, 14),
                       qmul=OpGenerator(4, operator.mul, 15),
                       omul=OpGenerator(8, operator.mul, 17),
                       fmul=OpGenerator(5, operator.mul, 18),
                       mul=OpGenerator(10, operator.mul, 16),))

generators.update(dict(baddz=OpGenerator(2, operator.add, 11, False),
                       qaddz=OpGenerator(4, operator.add, 12, False),
                       addz=OpGenerator(10, operator.add, 13, False),
                       bmulz=OpGenerator(2, operator.mul, 14, False),
                       qmulz=OpGenerator(4, operator.mul, 15, False),
                       mulz=OpGenerator(10, operator.mul, 16, False),))

class ToughAddGenerator(OpGenerator):
  def __init__(self, base, sep, zero_pad=True):
    super(ToughAddGenerator, self).__init__(base, operator.add, sep, zero_pad)

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
                       qaddt=ToughAddGenerator(4, 12),
                       addt=ToughAddGenerator(10, 13),))



class AlignedOpGenerator(OpGenerator):
  min_length = 2
  def rand_pair(self, l):
    k = int((l-1 - 2*PADDING)//2)
    n1, n2 = self._rand_inputs(k)
    result = self.f(n1, n2)
    n1, n2 = [np.concatenate([[START] if PADDING else [],
                              to_base(n, self.base, k) + 1,
                              #[22] if PADDING else []
                             ]) for n in [n1,n2]]
    preferred_length = l#max(len(n1), len(n2))+1
    pad_n1, pad_n2 = [np.pad(n,(0, preferred_length-len(n)), 'constant') for n in (n1, n2)]
    pad_n2[len(n2)] = self.sep
    inp2 = np.vstack([pad_n1, pad_n2])
    #XXX cheating on length here
    if True:
      o = np.concatenate([[START] if PADDING else [], to_base(result, self.base, l) + 1])
    else:
      o = to_base(result, self.base) + 1
      #o = to_base(result, self.base, k+2) + 1
    outp = np.pad(o, (0, preferred_length - len(o)), 'constant', constant_values=SPACE)
    return inp2, outp

class AlignedToughAddGenerator(AlignedOpGenerator, ToughAddGenerator):
  pass

generators.update(dict(badde=AlignedOpGenerator(2, operator.add, 11),
                       qadde=AlignedOpGenerator(4, operator.add, 12),
                       adde=AlignedOpGenerator(10, operator.add, 13),
                       bmule=AlignedOpGenerator(2, operator.mul, 14),
                       baddet=AlignedToughAddGenerator(2, 11),
                       qaddet=AlignedToughAddGenerator(4, 12),
                       addet=AlignedToughAddGenerator(10, 13),
                       baddzt=ToughAddGenerator(2, 11, False),
                       qaddzt=ToughAddGenerator(4, 12, False),
                       addzt=ToughAddGenerator(10, 13, False),
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


# With spacing
class SpacedGenerator(DataGenerator):
  height=4

  def is_valid_length(self, l):
    return super(SpacedGenerator, self).is_valid_length(l) and l >= self.min_length

  def rand_pair(self, l):
    l2 = np.random.randint(self.min_length, l)
    inp, res = self._rand_pair(l2)
    if isinstance(inp[0], int):
      inp = [inp]
    inp = np.array(inp)
    goal_dims = (self.height, l)
    bots = (0, 1 if PADDING else 0)
    tops = (goal_dims[0] - inp.shape[0], goal_dims[1] - inp.shape[1])
    placed_loc = [np.random.randint(b, t+1) for b, t in zip(bots, tops)]
    final_inp = np.zeros(goal_dims) + SPACE
    if PADDING:
      final_inp[:,0] = START
    final_inp[placed_loc[0]:placed_loc[0]+inp.shape[0],
              placed_loc[1]:placed_loc[1]+inp.shape[1]] = inp
    res = np.concatenate([res, [SPACE] * (l - len(res))])
    return (final_inp, res)

class CopyGenerator(SpacedGenerator):
  def __init__(self, base):
    self.base = base

  def _rand_pair(self, l):
    x = [np.random.randint(self.base)+1 for _ in xrange(l)]
    inp = x
    res = x
    return inp, res

class DupGenerator(SpacedGenerator):
  min_length = 2
  def __init__(self, base):
    self.base = base

  def _rand_pair(self, l):
    x = [np.random.randint(self.base)+1 for _ in xrange(l//2)]
    inp = [DUP] + x
    res = x + x
    return inp, res

class SpacedAlignedOpGenerator(SpacedGenerator, OpGenerator):
  def _rand_pair(self, l):
    k = int((l-1)//2)
    n1, n2 = self._rand_inputs(k)
    result = self.f(n1, n2)
    n1, n2 = [to_base(n, self.base) + 1 for n in [n1,n2]]
    preferred_length = max(len(n1), len(n2))
    inp = np.array([np.pad(n, (0, preferred_length - len(n)), 'constant',
                           constant_values=SPACE) for n in (n1, n2)])
    inp = np.concatenate([[[SPACE, self.sep]], inp.T]).T
    o = to_base(result, self.base) + 1
    return inp, o

class TSAOG(SpacedAlignedOpGenerator, ToughAddGenerator):
  pass

class SpacedOpGenerator(SpacedGenerator, OpGenerator):
  def _rand_pair(self, l):
    k = int((l-1)//2)
    n1, n2 = self._rand_inputs(k)
    result = self.f(n1, n2)
    n1, n2 = [to_base(n, self.base) + 1 for n in [n1,n2]]
    inp = np.concatenate([n1, [self.sep], n2])
    o = to_base(result, self.base) + 1
    return inp, o

class TSOG(SpacedOpGenerator, ToughAddGenerator):
  pass

generators.update(dict(scopy=CopyGenerator(10),
                       sdup=DupGenerator(10),
                       sbcopy=CopyGenerator(2),
                       sbdup=DupGenerator(2),
                       sbadde=SpacedAlignedOpGenerator(2, operator.add, 11),
                       sbmule=SpacedAlignedOpGenerator(2, operator.mul, 14),
                       sbaddet=TSAOG(2, 11),
                       sbadd=SpacedOpGenerator(2, operator.add, 11),
                       sbaddt=TSOG(2, 11),
                       sbaddz=SpacedOpGenerator(2, operator.add, 11, False),
                       sbaddzt=TSOG(2, 11, False),
                       sbmul=SpacedOpGenerator(2, operator.mul, 14),
                       ))


class MultiOpGenerator(DataGenerator):
  def __init__(self, base, f, sep, num, zero_chance=1, zero_pad=True):
    self.base = base
    self.f = f
    self.sep = sep
    self.num = num
    self.zero_pad = zero_pad
    self.min_length = 1 if num is None else 2*num - 1
    self.zero_chance = zero_chance

  def is_valid_length(self, l):
    return l >= self.min_length

  def _rand_inputs(self, k, num, allow_zero):
    k = int(k)
    return [random.randint(0 if allow_zero else 1, self.base**k-1) for i in range(num)]

  def rand_pair(self, l):
    num = self.num
    if num is None:
      num = random.randint(1, (l+1)//2)
    k = int((l+1)//num-1)
    allow_zero = random.random() < self.zero_chance
    ns = self._rand_inputs(k, num, allow_zero)
    result = functools.reduce(self.f, ns)
    input_arrays = []
    for i, n in enumerate(ns):
      if i:
        input_arrays.append([self.sep])
      input_arrays.append(to_base(n, self.base, k if self.zero_pad else 1)+1)
    inp = np.concatenate(input_arrays)
    outp = np.concatenate([
            to_base(result, self.base, (k+1)*num-1 if self.zero_pad else 1) + 1,
    ])
    return inp, outp

generators.update({'3badd':MultiOpGenerator(2, operator.add, 11, 3),
                   '3qadd':MultiOpGenerator(4, operator.add, 12, 3),
                   '3add':MultiOpGenerator(10, operator.add, 13, 3),
                   '3bmul':MultiOpGenerator(2, operator.mul, 14, 3),
                   })
generators.update({'kbadd':MultiOpGenerator(2, operator.add, 11, None),
                   'kqadd':MultiOpGenerator(4, operator.add, 12, None),
                   'kadd':MultiOpGenerator(10, operator.add, 13, None),
                   'kbmul':MultiOpGenerator(2, operator.mul, 14, None, .3),
                   })


for k in generators:
  generators[k].name = k

def set_height(self, height):
  for k in generators:
    generators[k].height = height







@np.vectorize
def char_to_symbol(i):
  """Covert ids to text."""
  i = int(i)
  if i == 0: return "_"
  if i in [11,12,13]: return "+"
  if i in [14,15,16]: return "*"
  if i in [START]: return '^'
  if i in [SPACE]: return '.'
  return str(i-1)

def join_array(array, rev=False):
  if len(array.shape) == 1:
    if rev:
      array = array[::-1]
    return ''.join(array).rstrip(' ')
  elif len(array.shape) == 2:
    if rev:
      array = array[:,::-1]
    return '\n'.join([''.join(a).rstrip(' ') for a in array])
  else:
    raise ValueError("Weird shape for joining: %s" % array.shape)

def to_string(array, rev=True):
  if isinstance(array, tuple):
    if len(array) == 3: # Batches
      inp, outp = array[:2]
      return '\n\n'.join(to_string((i,o), rev) for i,o in zip(inp, outp))
    inp, outp = [to_string(a, rev) for a in array[:2]]
    return '%s\n%s\n%s' % (inp, '-'*len(inp.split('\n')[0]), outp)
  return join_array(char_to_symbol(array), rev=rev)

@np.vectorize
def to_id(s):
  """Covert text to ids."""
  if s == "+": return 11
  if s == "*": return 14
  return int(s) + 1

class TeeErr(object):
    def __init__(self, f):
        self.file = f
        self.stderr = sys.stderr
        sys.stderr = self
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stderr.write(data)

log_f = None

def print_out(s, newline=True):
  """Print a message out and log it to file."""
  global log_f
  if log_filename:
    try:
      if log_f is None:
        log_f = open(log_filename, 'a', 1)
      log_f.write(s + ("\n" if newline else ""))
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
