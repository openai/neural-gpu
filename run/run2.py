import os
import sys
import collections

import argparse

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument('--label', type=str, default='Experiment')
parser.add_argument('paramoff', type=int, default=0)
parser.add_argument('gpuoff', type=int, default=0)


"""
all: rev
smoothing:
4: sigmoid + tanh
4: sigmoid
4: none

"""
PROGRAM='neural_gpu_trainer.py'

param_sets = [(('random_seed',seed),
               ('smooth_targets', st),
               )
              for st in [True]
              for seed in range(4)
              ]
param_sets = map(collections.OrderedDict, param_sets)

def to_name(params):
    return '-'.join([str(params[k]) for k in params])

'CUDA_VISIBLE_DEVICES=3 python neural_gpu_trainer.py --train_dir=logs/rev-nond5 --task rev  --smooth-grad=0.0'

def create_screen(session_label):
    os.system('screen -S %s -d -m' % (session_label,))

def run_with_options(gpu, screen_label, params, session_label=None):
    internal_command = 'CUDA_VISIBLE_DEVICES=%s python %s' % (gpu, PROGRAM)
    internal_command += ' ' + '--train_dir=logs/%s-%s' % (session_label, screen_label)
    internal_command += ' ' + ' '.join('--%s=%s' % vs for vs in params.items())
    screen_command = 'screen'
    screen_command += (' -S %s' % session_label if session_label else '')
    create_command = '%s -X screen -t "%s"' % (screen_command, screen_label)
    print create_command
    os.system(create_command)
    run_command = '%s -X -p "%s" stuff "%s\n"' % (screen_command, screen_label, internal_command)
    print run_command
    os.system(run_command)

def run_on_server(param_sets, session_label, gpuoff):
    if gpuoff == 0:
        create_screen(session_label)
    for i, params in enumerate(param_sets):
        name = to_name(params)
        run_with_options(i+gpuoff, name, params, session_label)

if __name__ == '__main__':
    args =  parser.parse_args()
    run_on_server(param_sets[args.paramoff:args.paramoff+8], args.label, args.gpuoff)
    
