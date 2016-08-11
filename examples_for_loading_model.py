import tensorflow as tf, neural_gpu_trainer, neural_gpu, neural_curriculum, data_utils, numpy as np
import glob
sess = tf.Session()


glob_txt = '../logs/June-13-6-withcutoffs/*seed0*/'
glob_txt = '../logs/June-14-01*/*seed4*/'
glob_txt = '../logs/June-14-07*/*badd-q*Fals*seed4*/'
glob_txt = '../logs/June-16-01-batchnorm/*batchnorm=True*seed4*/'
glob_txt = '../logs/June-22-03-clever/*baddet,*seed6*/'
ck_dir = glob.glob(glob_txt)[0]
model = neural_gpu_trainer.load_model(sess, ck_dir)
curriculum = neural_curriculum.Curriculum([data_utils.generators[x]
                        for x in 'badde'.split()], model.config)

example = data_utils.generators['baddet'].get_batch(8,32)

print data_utils.to_string(result.layer_outputs.argmax(axis=-1)[:,28,:70])

curriculum = neural_curriculum.DefaultCurriculum([data_utils.generators['badd']], model.config)

neural_gpu.FLAGS._parse_flags()
neural_gpu.FLAGS.do_attention = True
model = neural_gpu_trainer.initialize(sess)


tf.reset_default_graph();sess=tf.Session()
model = neural_gpu_trainer.initialize(sess) 

reload(neural_gpu);reload(neural_gpu_trainer);reload(neural_gpu.data_utils); neural_gpu.FLAGS.random_seed += 1

example = curriculum.draw_example(128)[0]
result = model.step(sess, example, False)

a=[0,2,10,0,2];model.step(sess, (np.array([a[::-1]+[-1,-1,-1]]).T+1, np.array([[2,1,1,1,1,0,0,0]]).T, [1]), False).output.argmax(axis=-1).T[0,:-3][::-1]-1
Out[266]: array([0, 0, 0, 1, 0])

def foo(model, sess, a):
    l = model.get_instance_for_length(len(a)).length
    pad = l - len(a)
    input = np.array([a[::-1] + [-1]*pad]).T + 1
    result = model.step(sess, (input, input, [0]), False)
    relevant_output = result.output.argmax(axis=-1).T[0, :(-pad if pad else None)]
    return relevant_output[::-1] - 1

    
