from __future__ import print_function
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

"""
all: rev
smoothing:
4: sigmoid + tanh
4: sigmoid
4: none

"""
USERNAME = os.environ['USER']
EXPERIMENT = 'neural-gpu'
basedir = os.path.dirname(os.path.realpath(__file__))


def to_str(params):
    return '-'.join(['%s=%s' % (k, params[k]) for k in params if k != 'random_seed'])


def to_name(params):
    options_str = to_str(params)
    return hashlib.sha224(options_str).hexdigest()[:10]


def run_with_options_commands(params):
    internal_command = args.program + ' ' + ' '.join('--%s=%s' % vs for vs in params.items())
    return internal_command


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
        metadata = yaml.load(f)
        names = metadata['names'].keys()
        experiment = metadata['experiment']
        user = metadata['user']

    for name in names:
        job_name = '{user}-{experiment}-{name}'.format(user=user,
                                                       experiment=experiment,
                                                       name=name)
        subprocess.check_call(['kubectl', 'delete', 'job', job_name])

    metadata['state'] = 'dead'
    with open(server_location, 'w') as f:
        f.write(yaml.safe_dump(metadata))
    print('Success! Writing state out to file.')


def run_opportunistically(param_sets, session_label):
    server_location = 'servers/%s' % session_label
    if os.path.isfile(server_location):
        raise ValueError('Server location file already exists!')

    with open(os.path.join(basedir, 'job.yaml.tpl'), 'r') as f:
        template = f.read()

    names = {}
    for params in param_sets:
        name = to_name(params)
        command = run_with_options_commands(params)
        job_filepath = os.path.join(basedir, 'deployed/{}.yaml'.format(name))
        with open(job_filepath, 'w') as f:
            f.write(template.format(experiment=EXPERIMENT,
                                    name=name, command=command,
                                    user=USERNAME,
                                    session_label=session_label))
        subprocess.check_call(['kubectl', 'create', '-f', job_filepath])
        names[name] = params

    metadata = {
        'experiment': EXPERIMENT,
        'user': USERNAME,
        'names': names,
        'label': session_label,
        'date': datetime.datetime.now(),
        'version': get_git_version(),
        'argv': sys.argv,
        'params': map(dict, param_sets),
        'state': 'alive'
    }

    with open(server_location, 'w') as f:
        f.write(yaml.safe_dump(metadata))

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


def main(param_sets):
    global args
    args = parser.parse_args()
    if not args.kill and check_git_modified():
        if args.force:
            print('Continuing anyway...')
        else:
            print('Please commit first.')
            return
    if not args.local:
        if args.kill:
            kill(args.label)
            print("Repeating, for good measure")
            kill(args.label)
            return
        run_opportunistically(param_sets, args.label)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main([{'do_batchnorm': 0, 'task': 'scopy,sdup', 'progressive_curriculum': True, 'do_outchoice': True}])
