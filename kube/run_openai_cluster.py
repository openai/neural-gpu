import argparse
import datetime
import hashlib
import os
import os.path
import sys
import subprocess
import yaml

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument('--label', type=str, default='Experiment')
parser.add_argument('--dry', action='store_true', default=False)
parser.add_argument('--local', action='store_true', default=False)
parser.add_argument('--kill', action='store_true', default=False)
parser.add_argument('--force', action='store_true', default=False)
parser.add_argument('--program', type=str, default='python neural_gpu_trainer.py')
parser.add_argument('paramoff', nargs='?', type=int, default=0)
parser.add_argument('gpuoff', nargs='?', type=int, default=0)

EXPERIMENT = 'neural-gpu'

def short_name(params):
    return hashlib.sha224(str(params)).hexdigest()[:10]

def run_with_options_commands(params):
    internal_command = args.program + ' ' + ' '.join('--%s=%s' % vs for vs in params.items())
    return internal_command

def start_job(param_sets,label):
    commands = [run_with_options_commands(params) for params in param_sets]
    subprocess.check_call(['openai-cluster','-vv','start_batch','-l',label,'--num-gpus','1']+commands)

def main(param_sets):
    global args
    args = parser.parse_args()

    if args.kill or args.dry or args.local or args.force:
        raise NotImplementedError()

    start_job(param_sets, args.label)

if __name__ == '__main__':
    main([{'do_batchnorm': 0, 'task': 'scopy,sdup', 'progressive_curriculum': True, 'do_outchoice': True}])
