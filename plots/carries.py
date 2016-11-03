"""Class for constructing problem inputs featuring lots of carries."""
from __future__ import print_function

import tensorflow as tf, numpy as np

import operator
import pandas
import random
import time
import glob
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neuralgpu import generators



def get_generator(base, sep, aligned=False, randloc=False):
    base_class = generators.AlignedOpGenerator if aligned else generators.OpGenerator
    class CarryGenerator(base_class):
        def __init__(self, carry, overflow, randloc=randloc, base=base, sep=sep, zero_pad=True):
            super(CarryGenerator, self).__init__(base, operator.add, sep)
            self.carry = carry
            self.overflow = overflow
            self.randloc = randloc

        def _rand_inputs(self, k):
            n1 = random.randint(1 if self.overflow else 0, self.base**self.carry-1)
            n2 = self.base**self.carry - n1 - (0 if self.overflow else 1)
            loc = random.randint(0, k - self.carry) if self.randloc else 0
            vals = [n1*self.base**loc, n2*self.base**loc]
            if random.random() > .5:
                return vals
            else:
                return vals[::-1]

        @classmethod
        def get_error_rate(cls, sess, model, carry_length, do_overflow, max_length, num):
            if max_length is None:
                max_length = 2*carry_length + 3
            example = cls(carry_length, do_overflow).get_batch(max_length, num)
            result = model.step(sess, example, False)
            return result.accuracy()[2]

        @classmethod
        def get_rates(cls, sess, model, carries, max_length=201, numblocks=1, blocksize=32, verbose=True):
            df = pandas.DataFrame(index=carries, columns=[False, True])
            for carry in carries:
                for col in df.columns:
                    ans = 0
                    for i in range(numblocks):
                        ans += cls.get_error_rate(sess, model, carry, col, max_length, blocksize)
                    df[col][carry] = ans
                if verbose:
                    print(carry, ':', df[False][carry], df[True][carry])
            return df

    return CarryGenerator
