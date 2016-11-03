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
"""Utilities for the NeuralGPU

This file has two main components:

 - generators is a dict mapping task names to DataGenerator instances, which construct individual problem input/output pairs
 - Utilities for converting those input/output pairs to/from string representations.
"""

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

# Lengths of NeuralGPU instances.  Inputs will be padded to the next
# larger one.
bins = [8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64, 128]
forward_max = 128
log_filename = ""


DIGITS = range(1, 11)
NULL = 0
DUP = 22
SPACE = 23
START = 24
MINUS = 25

def pad(l):
  for b in bins + [forward_max]:
    if b >= l: return b
  raise IndexError("Length %s longer than max length %s" % (l, forward_max))


@np.vectorize
def char_to_symbol(i):
  """Covert ids to text."""
  i = int(i)
  if i == 0: return "_"
  if i in [11,12,13]: return "+"
  if i in [14,15,16]: return "*"
  if i in [17,18,19]: return "/"
  if i in [20,21,22]: return "-"
  if i in [START]: return '^'
  if i in [SPACE]: return '.'
  if i in [MINUS]: return '-'
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
