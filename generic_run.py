import os
import sys
import collections
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument('--label', type=str, default='Experiment')
parser.add_argument('--dry', action='store_true', default=False)
parser.add_argument('--on_servers', action='store', default='')
parser.add_argument('--kill', action='store_true', default=False)
parser.add_argument('--force', action='store_true', default=False)
parser.add_argument('paramoff', nargs='?', type=int, default=0)
parser.add_argument('gpuoff', nargs='?', type=int, default=0)

"""
all: rev
smoothing:
4: sigmoid + tanh
4: sigmoid
4: none

"""
PROGRAM='neural_gpu_trainer.py'

def to_name(params):
    return '-'.join(['%s=%s' % (k, params[k]) for k in params if k != 'random_seed'])

def create_screen_commands(session_label):
    return ['screen -S %s -d -m' % (session_label,)]

def run_with_options_commands(gpu, screen_label, params, session_label=None):
    internal_command = 'CUDA_VISIBLE_DEVICES=%s python %s' % (gpu, PROGRAM)
    internal_command += ' ' + '--train_dir=../logs/%s-%s' % (session_label, screen_label)
    internal_command += ' ' + ' '.join('--%s=%s' % vs for vs in params.items())
    screen_command = 'screen'
    screen_command += (' -S %s' % session_label if session_label else '')
    create_command = '%s -X screen -t "%s"' % (screen_command, screen_label)
    command = '%s -X -p "%s" stuff "%s\n"' % (screen_command, screen_label, internal_command)
    return [create_command, command]

def oneserver_commands(param_sets, session_label, gpuoff):
    commands = []
    if gpuoff == 0:
        commands.extend(create_screen_commands(session_label))
    for i, params in enumerate(param_sets):
        name = to_name(params)
        commands.extend(run_with_options_commands(i+gpuoff, name, params, session_label))
    return commands

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
    ssh_cmd = ["ssh", "%s.cirrascale.sci.openai-tech.com" % server,
               "cd models/neural_gpu\n%s" % '\n'.join(commands)]
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
    if args.on_servers:
        assert args.gpuoff == 0
        servers = args.on_servers.split(',')
        if args.kill:
            for s in servers:
                run_remotely(s, ['sh kill_screen.sh'])
            return
        if len(servers) * 8 < len(param_sets):
            print 'Not enough servers! %s/%s > 8' % (len(param_sets), len(servers))
            if args.force:
                print 'Continuing anyway'
            else:
                return
        servers_used = False
        for i in range((len(param_sets) + 7)//8):
            pids = check_server_usage(servers[i])
            if pids:
                print 'Error: server %s already has gpus used by %s' % (servers[i], pids)
                servers_used = True
        if servers_used:
            if args.force:
                print 'Continuing anyway'
            else:
                return

        for i, s in enumerate(servers):
            to_run = param_sets[args.paramoff+8*i:][:8]
            if not to_run:
                continue
            commands = oneserver_commands(to_run, args.label, args.gpuoff)
            run_remotely(s, commands)
        print 'Ran %s/%s jobs' % (len(param_sets[args.paramoff:][:8*len(servers)]), len(param_sets))
    else:
        to_run = param_sets[args.paramoff:][:8]
        commands = oneserver_commands(to_run, args.label, args.gpuoff)
        run_here(commands)
        print 'RAN %s:%s of %s' % (args.paramoff, args.paramoff+8, len(param_sets))
