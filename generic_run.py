import os
import sys
import collections
import subprocess
import argparse

import cirrascale.client
from requests import HTTPError

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument('--label', type=str, default='Experiment')
parser.add_argument('--dry', action='store_true', default=False)
parser.add_argument('--local', action='store_true', default=False)
parser.add_argument('--kill', action='store_true', default=False)
parser.add_argument('--force', action='store_true', default=False)
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

def find_free_gpus(num_needed):
    all_gpu_status = cirrascale.client.get_gpu_status()
    free_gpu_status = {k:[g for g in v if g.available]
                       for k,v in all_gpu_status.items()}
    num_free = sum(map(len, free_gpu_status.values()))
    print '%s/%s GPUS' % (num_needed, num_free)
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
            print 'Error reserving the GPUs; race? Trying again.'
        else:
            break
    return choices


def to_name(params):
    return '-'.join(['%s=%s' % (k, params[k]) for k in params if k != 'random_seed'])

def create_screen_commands(session_label):
    return ['screen -S %s -d -m' % (session_label,)]

def run_with_options_commands(gpu, screen_label, params, session_label=None):
    internal_command = 'CUDA_VISIBLE_DEVICES=%s %s' % (gpu, args.program)
    log_dir = '../logs/%s' % session_label
    internal_command += ' ' + '--train_dir=%s/%s' % (log_dir, screen_label)
    internal_command += ' ' + ' '.join('--%s=%s' % vs for vs in params.items())
    screen_command = 'screen'
    screen_command += (' -S %s' % session_label if session_label else '')
    create_command = '%s -X screen -t "%s"' % (screen_command, screen_label)
    command = '%s -X -p "%s" stuff "%s\n"' % (screen_command, screen_label, internal_command)
    result = [create_command,
              'mkdir -p %s' % log_dir,
              command]
    return result

def oneserver_commands(param_sets, session_label, gpus):
    commands = []
    commands.extend(create_screen_commands(session_label))
    for gpu, params in zip(gpus, param_sets):
        name = to_name(params)
        commands.extend(run_with_options_commands(gpu.index, name, params, session_label))
    return commands

def kill(session_label):
    server_location = 'servers/%s' % session_label
    with open(server_location) as f:
        gpudict = {}
        for line in f:
            host = line.split()[0]
            indices = line.split()[1:]
            gpudict[host] = set(indices)
    for s in gpudict:
        run_remotely(s, ['sh kill_screen.sh %s' % session_label])
    all_gpus = cirrascale.client.get_gpu_status()
    to_unreserve = []
    for g in sum(all_gpus.values(), []):
        if g.reserved and g.reserved['username'] == USERNAME:
            if g.host in gpudict and g.index in gpudict[g.host]:
                to_unreserve.append(g.id)
    cirrascale.client.release_gpus(to_unreserve)

def run_opportunistically(param_sets, session_label):
    server_location = 'servers/%s' % session_label
    if os.path.isfile(server_location):
        raise ValueError('Server location file already exists!')
    gpudict = grab_gpus(len(param_sets))
    print 'Got GPUs:', gpudict
    with open(server_location, 'w') as f:
        for h, lst in gpudict.items():
            print >> f, h, ' '.join(g.index for g in lst)
    done = 0
    for h, gpus in gpudict.items():
        commands = oneserver_commands(param_sets[done:done+len(gpus)],
                                      session_label, gpus)
        done += len(gpus)
        print h, commands
        run_remotely(h, commands)


def check_git_modified():
    files = subprocess.check_output(['git', 'ls-files', '-m'])
    if files:
        print 'ERROR: modified files:'
        for f in files.splitlines():
            print '  '+f.strip()
        return True
    return False

def check_server_usage(server):
    command = 'lsof -nt /dev/nvidia*'
    procs = run_remotely(server, [command, 'true'])
    return procs.split()

def run_remotely(server, commands):
    print 'ON %s:' % server
    ssh_cmd = ["ssh", server,
               "cd %s\n%s" % (args.dir, '\n'.join(commands))]
    print '\n'.join(commands)
    outp = subprocess.check_output(ssh_cmd)
    print 'DONE!'
    return outp

def run_here(commands):
    for c in commands:
        print cmd
        if not args.dry:
            os.system(cmd)


def main(param_sets):
    global args
    args =  parser.parse_args()
    if not args.kill and check_git_modified():
        if args.force:
            print 'Continuing anyway...'
        else:
            print 'Please commit first.'
            return
    if not args.local:
        if args.kill:
            kill(args.label)
            return
        run_opportunistically(param_sets, args.label)
    else:
        to_run = param_sets[args.paramoff:][:8]
        commands = oneserver_commands(to_run, args.label, range(args.gpuoff, 8))
        run_here(commands)
        print 'RAN %s:%s of %s' % (args.paramoff, args.paramoff+8, len(param_sets))
