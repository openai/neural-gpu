"""Compute statistics on model checkpoints and long carries in decimal addition.


It looks for checkpoints of the form "../logs/September-*/*/neural_gpu.ckpt-100000

When run with different arguments, computes different statistics which
are placed in different files in the checkpoint directory; if that
file already exists, it does not compute the file.  Hence you can
repeatedly run this program, as you create more checkpoints.

With no arguments, in 'carries.csv' it places the success rate for various lengths of carries.
With '-t', in 'thresholds' it places the minimum threshold at which the success rate is < 50%
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import operator
import pandas
import random
import time
import os
import glob
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neuralgpu import trainer, data_utils
import carries

#data_utils.bins.pop()
#data_utils.bins.pop()

#del data_utils.bins[6]
#del data_utils.bins[4]

# Because of the bug with 'tf.Variable' rather than 'tf.get_variable' for 'layer_index' in neural_gpu.py,
# we need to have an equal number of bins to when it was trained.
data_utils.bins = [8] + [2**i + 5 for i in range(3, 13)]

aligned = False
base, sep = (10, 13)
randloc = False
CarryGenerator = carries.get_generator(base, sep, aligned, randloc)

dir = None
model = None
sess = None

def load_model(dir):
    global model, sess
    reconfig = {'mode': 1,           # No backprop
                'forward_max': 401}  # Large enough to check 200-digit carries
    if model is None:
        sess=tf.Session()
        model = trainer.load_model(sess, dir, reconfig)
    else:
        model.saver.restore(sess, dir+'/neural_gpu.ckpt-100000')


def find_dirs(base_dir='../logs', check_file='carries.csv'):
    """Find all checkpoint directories that haven't been updated for check_file"""
    for one in glob.glob(base_dir+'/September-*'):
        for full_dir in glob.glob(one+'/*'):
            if os.path.exists(full_dir+'/neural_gpu.ckpt-100000'):
                if not os.path.exists(os.path.join(full_dir, check_file)):
                    yield full_dir


locs = list(range(1, 30)) + list(range(30,100,5))

def get_data(dir, locs=locs):
    load_model(dir)
    results = CarryGenerator.get_rates(sess, model, locs, 201 if randloc else None, 1)
    return results

def run_dir(dir):
    try:
        results = get_data(dir)
    except tensorflow.python.framework.errors.FailedPreconditionError as e:
        print('ERROR ON DIR', dir, file=sys.stderr)
        print()
        print(e)
        print()
        return
    with open(dir+'/carries.csv', 'w') as f:
        f.write(results.to_csv())

def bsearch(is_leq, lo=1, hi=None):
    if hi is None:
        hi =  2*lo
        while not is_leq(hi):
            lo, hi = hi+1, 2*hi
    while lo < hi:
        mid = (lo+hi)//2
        if is_leq(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def find_threshold():
    def is_leq(n):
        def get_estimate(blocksize):
            return sum(CarryGenerator.get_error_rate(sess, model, n, truth, None, blocksize) for truth in [False, True]) * 1./(2*blocksize)

        blocksize = 32
        result = get_estimate(blocksize)
        print(n, result)
        # Be extra careful once we get close
        if abs(result - .5) < .2:
            result = np.mean([result] + [get_estimate(blocksize) for _ in range(2)])
            print('Refined estimate:', result)
        return result >= .5
    return bsearch(is_leq)

def main_results():
    for dir in find_dirs():
        print('Checking', dir)
        run_dir(dir)

def main_thresholds(fname = 'threshold'):
    for dir in find_dirs(check_file=fname):
        print('Checking', dir)
        load_model(dir)
        thresh = find_threshold()
        with open(os.path.join(dir, fname), 'w') as f:
            print(thresh, file=f)

if __name__ == '__main__':
    if '-t' in sys.argv:
        main_thresholds()
    else:
        main_results()
