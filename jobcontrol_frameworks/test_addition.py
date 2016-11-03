"""Train many instances of neural GPU on decimal addition, with different random seeds"""
from __future__ import print_function
import argparse
import os
import subprocess
import sys
import shutil

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--count', type=int, default=100)
args =  parser.parse_args()



BASE_SEED = 1000000 * args.seed

basedir = '../logs/September-02-{}'.format(BASE_SEED)
train_dir_fmt = basedir + '/forward_max=201-nmaps=128-task=add-seed={num}'
cmd_fmt = 'python neural_gpu_trainer.py --train_dir={train_dir} --random_seed={seed} --max_steps=100000 --forward_max=201 --nmaps=128 --task=add --time_till_eval=4 --time_till_ckpt=30'

def run_once(num):
    vars = dict(base_seed=BASE_SEED, num=num, seed=BASE_SEED + num)
    vars['train_dir'] = train_dir_fmt.format(**vars)
    if os.path.exists('%s/neural_gpu.ckpt-100000' % vars['train_dir']):
        print('Already completed, ignoring')
        return
    if os.path.exists(vars['train_dir']):
        shutil.rmtree(vars['train_dir'])
        
    cmdline = cmd_fmt.format(**vars)
    print("Running", num)
    result = subprocess.call(cmdline, shell=True)
    return result

if not os.path.exists(basedir):
    os.mkdir(basedir)

for j in range(args.count):
    run_once(j)
