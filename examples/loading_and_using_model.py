import numpy as np
import tensorflow as tf

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neuralgpu import trainer, data_utils

sess = tf.Session()
ck_dir = '/tmp/moo/cow2'
model = trainer.load_model(sess, ck_dir)

example = data_utils.generators['baddet'].get_batch(8,32)

result = model.step(sess, example, False)
print(result.to_string())
