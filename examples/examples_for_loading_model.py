from __future__ import print_function
import tensorflow as tf, neural_gpu_trainer, neural_gpu, neural_curriculum, data_utils, numpy as np
sess = tf.Session()


ck_dir = '/log/dir'
model = neural_gpu_trainer.load_model(sess, ck_dir)

example = data_utils.generators['baddet'].get_batch(8,32)

result = model.step(sess, example, False)
print(result.to_string())
