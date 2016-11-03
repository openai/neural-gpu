"""Generators for the different problems."""

import math
import random
import sys
import time
import operator
import functools
import numpy as np

from . import data_utils
from .data_utils import SPACE, START, MINUS, DUP

# This maps task names to DataGenerator instances
generators = {}

PADDING = False

def to_base(num, b, l=1):
  if num < 0:
    val = to_base(-num, b, (l - 1) or 1)
    return np.concatenate([val, [MINUS-1]])
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


class DataGenerator(object):
  """The base class for generating problem input/output pairs"""
  nclass = 33
  name = '<unknown task>'
  taskid = 0
  height = None
  min_length = 1

  def is_valid_length(self, l):
    """Can this problem have instances of length l?"""
    return True

  def rand_pair(self, length):
    """Random data pair for a task. Total length should be <= length."""
    raise NotImplementedError()

  def rand_pair_padded(self, length):
    """Construct a random data pair, then pad the inputs to a valid size."""
    pad_length = data_utils.pad(length)
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
    """Construct a complete batch of problem instances"""
    inps, outps = [], []
    for _ in range(batch_size):
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
  """Generator for instances using operations on two variables in some base"""
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
  """More adversarial inputs for addition"""
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
  """Two-line binary inputs"""
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
    o = np.concatenate([[START] if PADDING else [], to_base(result, self.base, l) + 1])
    outp = np.pad(o, (0, preferred_length - len(o)), 'constant', constant_values=SPACE)
    return inp2, outp

class AlignedToughAddGenerator(AlignedOpGenerator, ToughAddGenerator):
  pass

generators.update(dict(badde=AlignedOpGenerator(2, operator.add, 11),
                       qadde=AlignedOpGenerator(4, operator.add, 12),
                       adde=AlignedOpGenerator(10, operator.add, 13),
                       bmule=AlignedOpGenerator(2, operator.mul, 14),
                       qmule=AlignedOpGenerator(4, operator.mul, 15),
                       mule=AlignedOpGenerator(10, operator.mul, 16),
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
    if not hasattr(inp[0], '__iter__'):
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
    x = [np.random.randint(self.base)+1 for _ in range(l)]
    inp = x
    res = x
    return inp, res

class DupGenerator(SpacedGenerator):
  min_length = 2
  def __init__(self, base):
    self.base = base

  def _rand_pair(self, l):
    x = [np.random.randint(self.base)+1 for _ in range(l//2)]
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
  """Inputs where a single operation can appear many times"""
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

class ExpressionGenerator(DataGenerator):
  """Inputs where each character has a chance of being a random operator."""
  min_length = 1

  def __init__(self, base, operators, op_chance):
    self.base = base
    self.operators = dict(operators)
    self.nums = range(base)
    self.op_chance = op_chance

    self.to_num = {i: i+1 for i in self.nums}
    self.to_num.update(self.operators)

  def rand_pair(self, l):
    ans = []
    inp = []
    last_num = []
    valid_op = False
    for i in range(l):
      if valid_op and random.random() < self.op_chance:
        choice = random.choice(self.operators.keys())
      else:
        choice = random.choice(self.nums)
      inp.append(self.to_num[choice])
      if choice in self.operators:
        ans.append(from_base(last_num, self.base))
        last_num = []
        ans.append(choice)
        valid_op = False
      else:
        last_num.append(choice)
        if i == l-2:
          valid_op = False
        else:
          valid_op = True
    ans.append(from_base(last_num, self.base))
    string_expr = ''.join(map(str, ans[::-1]))
    string_expr = string_expr.replace('/', '//')
    try:
      result = eval(string_expr)
    except ZeroDivisionError:
      return self.rand_pair(l)
    if result < 0:
      return self.rand_pair(l)
    outp = to_base(result, self.base, l)+1
    return inp, outp

generators.update({'bexpr':ExpressionGenerator(2, zip('+*', [11, 14]), .3),
                   'qexpr':ExpressionGenerator(4, zip('+*', [12, 15]), .3),
                   'expr':ExpressionGenerator(10, zip('+*', [13, 16]), .3),})

generators.update({'bexpra':ExpressionGenerator(2, zip('+*/-', [11, 14,17,20]), .3),
                   'qexpra':ExpressionGenerator(4, zip('+*/-', [12, 15,18,21]), .3),
                   'expra':ExpressionGenerator(10, zip('+*/-', [13, 16,19,22]), .3),})

generators.update({'bexprp':ExpressionGenerator(2, zip('+', [11]), .3),
                   'qexprp':ExpressionGenerator(4, zip('+', [12]), .3),
                   'exprp':ExpressionGenerator(10, zip('+', [13]), .3),})

generators.update({'bexprs':ExpressionGenerator(2, zip('+-', [11, 20]), .3),
                   'qexprs':ExpressionGenerator(4, zip('+-', [12, 21]), .3),
                   'exprs':ExpressionGenerator(10, zip('+-', [13, 22]), .3),})

generators.update({'bexprsm':ExpressionGenerator(2, zip('+*-', [11, 14,20]), .3),
                   'qexprsm':ExpressionGenerator(4, zip('+*-', [12, 15,21]), .3),
                   'exprsm':ExpressionGenerator(10, zip('+*-', [13, 16,22]), .3),})

for k in generators:
  generators[k].name = k

def set_height(self, height):
  for k in generators:
    generators[k].height = height

