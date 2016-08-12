import tensorflow as tf, neural_gpu_trainer, neural_gpu, neural_curriculum, data_utils, numpy as np
import glob
sess = tf.Session()


glob_txt = '../logs/June-22-03-clever/*baddet,*seed6*/'
ck_dir = glob.glob(glob_txt)[0]
model = neural_gpu_trainer.load_model(sess, ck_dir)

example = data_utils.generators['baddet'].get_batch(8,32)

result = model.step(sess, example, False)
print result.to_string()
