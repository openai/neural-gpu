from __future__ import print_function
import os
import sys
import collections
import subprocess
import argparse
import yaml
import datetime

import cirrascale.client
from requests import HTTPError

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument('--label', type=str, default='Experiment')
parser.add_argument('--session', type=str, default='')
parser.add_argument('--dry', action='store_true', default=False)
parser.add_argument('--local', action='store_true', default=False)
parser.add_argument('--kill', action='store_true', default=False)
parser.add_argument('--force', action='store_true', default=False)
parser.add_argument('--use_full', action='store_true', default=False)
parser.add_argument('--dir', type=str, default='models/neural_gpu')
parser.add_argument('--program', type=str, default='python neural_gpu_trainer.py')
parser.add_argument('paramoff', nargs='?', type=int, default=0)
parser.add_argument('gpuoff', nargs='?', type=int, default=0)

"""
all: rev
smoothing:
4: sigmoid + tanh
4: sigmoid
4: none

"""
USERNAME = os.environ['USER']
DROPPED_KEYS = ('random_seed', 'max_steps', 'time_till_eval')

def find_free_gpus(num_needed):
    all_gpu_status = cirrascale.client.get_gpu_status()
    free_gpu_status = {k:[g for g in v if g.available]
                       for k,v in all_gpu_status.items()}
    if not args.use_full:
        free_gpu_status = {k:v for k,v in free_gpu_status.items()
                           if len(v) < 8}
    num_free = sum(map(len, free_gpu_status.values()))
    print('%s/%s GPUS' % (num_needed, num_free))
    if num_free < num_needed:
        raise ValueError("Insufficient GPUs available!")
    choices = {}
    while num_needed:
        free_counts = {k:len(v) for k, v in free_gpu_status.items()}
        next_k = max(free_counts, key=lambda k: 100-free_counts[k] if free_counts[k] >= num_needed else free_counts[k])
        choices[next_k] = free_gpu_status[next_k][:num_needed]
        num_needed -= len(choices[next_k])
        del free_gpu_status[next_k]
    return choices

def grab_gpus(num_needed, t=5*60):
    while True:
        choices = find_free_gpus(num_needed)
        gpu_strings = [g.id for v in choices.values() for g in v]
        try:
            cirrascale.client.reserve_gpus(USERNAME, gpu_strings, t)
        except HTTPError as e:
            print('Error reserving the GPUs; race? Trying again.')
        else:
            break
    return choices


def to_name(params):
    return '-'.join(['%s=%s' % (k, params[k]) for k in params if k not in DROPPED_KEYS])

def create_screen_commands(session_label):
    return ['screen -S %s -d -m' % (session_label,)]

def get_train_dir(screen_label, session_label):
    return '../logs/%s/%s' % (session_label, screen_label)

def run_with_options_commands(gpu, screen_label, params, session_label=None):
    internal_command = 'CUDA_VISIBLE_DEVICES=%s %s' % (gpu, args.program)
    log_dir = get_train_dir(screen_label, session_label)
    internal_command += ' ' + '--train_dir=%s' % log_dir
    internal_command += ' ' + ' '.join('--%s=%s' % vs for vs in params.items())
    screen_command = 'screen'
    screen_command += (' -S %s' % session_label if session_label else '')
    create_command = '%s -X screen -t "%s"' % (screen_command, screen_label)
    command = '%s -X -p "%s" stuff "%s\n"' % (screen_command, screen_label, internal_command)
    result = [create_command,
              'mkdir -p %s' % os.path.dirname(log_dir),
              command]
    return result

def oneserver_commands(param_sets, session_label, gpus):
    commands = []
    commands.extend(create_screen_commands(session_label))
    for gpu, params in zip(gpus, param_sets):
        name = to_name(params)
        commands.extend(run_with_options_commands(gpu.index, name, params, session_label))
    return commands

def kill(session_label, server_file):
    server_location = 'servers/%s' % (server_file or session_label)
    with open(server_location) as f:
        metadata = yaml.load(f)
        gpudict = metadata['locations']
    for s in gpudict:
        run_remotely(s, ['sh kill_screen.sh %s' % session_label])
    all_gpus = cirrascale.client.get_gpu_status()
    to_unreserve = []
    for g in sum(all_gpus.values(), []):
        if g.reserved and g.reserved['username'] == USERNAME:
            if g.host in gpudict and g.index in gpudict[g.host]:
                to_unreserve.append(g.id)
    cirrascale.client.release_gpus(to_unreserve)
    metadata['state'] = 'dead'
    with open(server_location, 'w') as f:
        f.write(yaml.safe_dump(metadata))
    print('Success! Writing state out to file.')

def run_opportunistically(param_sets, session_label, server_file=None):
    server_file = server_file or session_label
    server_location = 'servers/%s' % server_file
    if os.path.isfile(server_location):
        raise ValueError('Server location file already exists!')
    gpudict = grab_gpus(len(param_sets))
    alt_gpudict = {h : [g.index for g in lst] for (h, lst) in gpudict.items()}
    metadata = dict(locations = alt_gpudict,
                  label = session_label,
                  date = datetime.datetime.now(),
                  version = get_git_version(),
                  argv = sys.argv,
                  params = map(dict, param_sets),
                  state = 'alive'
                  )
    print('Got GPUs:')
    for k in gpudict:
        print(k, len(gpudict[k]))
    with open(server_location, 'w') as f:
        f.write(yaml.safe_dump(metadata))
    done = 0
    for h, gpus in sorted(gpudict.items()):
        commands = oneserver_commands(param_sets[done:done+len(gpus)],
                                      session_label, gpus)
        done += len(gpus)
        run_remotely(h, commands)
    print('Done')


def get_git_version():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

def check_git_modified():
    files = subprocess.check_output(['git', 'ls-files', '-m'])
    if files:
        print('ERROR: modified files:')
        for f in files.splitlines():
            print('  '+f.strip())
        return True
    return False

def check_server_usage(server):
    command = 'lsof -nt /dev/nvidia*'
    procs = run_remotely(server, [command, 'true'])
    return procs.split()

def run_remotely(server, commands):
    print('Running commands %s.' % server)
    ssh_cmd = ["ssh", server,
               "cd %s\n%s" % (args.dir, '\n'.join(commands))]
    #print('\n'.join(commands))
    outp = subprocess.check_output(ssh_cmd)
    #print('DONE!')
    return outp

def run_here(commands):
    for c in commands:
        print(cmd)
        if not args.dry:
            os.system(cmd)


def main(param_sets):
    global args
    args =  parser.parse_args()
    if not args.kill and check_git_modified():
        if args.force:
            print('Continuing anyway...')
        else:
            print('Please commit first.')
            return
    if not args.local:
        if args.kill:
            kill(args.label, args.session)
            print("Repeating, for good measure")
            kill(args.label, args.session)
            return
        run_opportunistically(param_sets, args.label, args.session)
    else:
        to_run = param_sets[args.paramoff:][:8]
        commands = oneserver_commands(to_run, args.label, range(args.gpuoff, 8))
        run_here(commands)
        print('RAN %s:%s of %s' % (args.paramoff, args.paramoff+8, len(param_sets)))
