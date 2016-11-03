import numpy as np
import tensorflow as tf

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neuralgpu import trainer, generators

DIR = '/tmp/moo/cow3'

sess = tf.Session()
model = trainer.load_model(sess, DIR)

example = generators.generators['baddet'].get_batch(8,32)

result = model.step(sess, example, False)
print(result.to_string())
