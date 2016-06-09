import os
import sys

"""
all: rev
smoothing:
4: sigmoid + tanh
4: sigmoid
4: none

"""
PROGRAM='neural_gpu_trainer.py'


param_sets = [dict(random_seed=seed,
                   smooth_grad=sigmoid_cutoff,
                   smooth_grad_tanh=tanh_cutoff,
                   )
              for sigmoid_cutoff, tanh_cutoff in [(0, 0), (10, 0),
                                                  (10,10)]
              for seed in range(4)
              ]

def to_name(params):
    return '%s-%s-%s' % (params['random_seed'], params['smooth_grad'], params['smooth_grad_tanh'])

'CUDA_VISIBLE_DEVICES=3 python neural_gpu_trainer.py --train_dir=logs/rev-nond5 --task rev  --smooth-grad=0.0'

def create_screen(session_label):
    os.system('screen -S %s -d -m' % (session_label,))

def run_with_options(gpu, screen_label, params, session_label=None):
    internal_command = 'CUDA_VISIBLE_DEVICES=%s python %s' % (gpu, PROGRAM)
    internal_command += ' ' + '--train_dir=logs/%s-%s' % (session_label, screen_label)
    internal_command += ' ' + ' '.join('--%s=%s' % vs for vs in params.items())
    screen_command = 'screen'
    screen_command += (' -S %s' % session_label if session_label else '')
    os.system('%s -X screen -t "%s"' % (screen_command, screen_label))
    os.system('%s -X stuff "%s\n"' % (screen_command, internal_command))

def run_on_server(param_sets, session_label):
    create_screen(session_label)
    for i, params in enumerate(param_sets):
        name = to_name(params)
        run_with_options(i, name, params, session_label)

if __name__ == '__main__':
    offset = int(sys.argv[1])
    run_on_server(param_sets[offset:offset+8], 'Experiment')
    
