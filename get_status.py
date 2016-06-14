from __future__ import print_function
import glob
import subprocess
import collections
import sys

import yaml
import numpy as np

import click

def get_fnames():
    return glob.glob('servers/*')

def get_states():
    for fname in get_fnames():
        with open(fname) as f:
            yield yaml.load(f)

def load_log(fname):
    try:
        lines = open(fname).readlines()
    except IOError:
        return None
    step_lines = [x for x in lines if x.startswith('step ')]
    if not step_lines:
        return lines[-1].strip()
    else:
        data = step_lines[-1].split()
        return collections.OrderedDict(zip(data[::2], data[1::2]))

def locked_on_server(server):
    cmd = 'python models/neural_gpu/used_gpus.py'
    if sys.version_info.major == 3:
        try:
            output = subprocess.check_output(['ssh', server, cmd], timeout=5)
            output = output.decode()
        except subprocess.TimeoutExpired:
            print('ERROR! %s dead.' % server)
            return []
    else:
        output = subprocess.check_output(['ssh', server, cmd], timeout=5)
    return output.split()

class Results(object):
    def __init__(self, metadata):
        self.__dict__.update(metadata)
        self.metadata = metadata

    def _parse_logs(self):
        dirs = glob.glob('logs/%s/*' % self.label)
        self.results = [load_log('%s/log0' % dir) for dir in dirs]

    def _running_programs(self):
        self.dead = {}
        for server in self.locations:
            still_running = locked_on_server(server)
            dead = [x for x in self.locations[server] if x not in still_running]
            if dead:
                self.dead[server] = dead

    @property
    def status(self):
        vals = [res for res in self.results if isinstance(res, dict)]
        if vals:
            answer = collections.OrderedDict()
            for key in vals[0]:
                answer[key] = np.median([float(v[key]) for v in vals])
            formatting = dict(step='%d', len='%d')
            return ' '.join('%s %s' % (key,
                                       formatting.get(key, '%0.3g') % v) for key, v in answer.items())
        else:
            if not self.results:
                return 'No output found!'
            else:
                return self.results[0]

    @property
    def accuracy(self):
        return '%0.2f' % np.median([float(r['sequence-errors']) for r in self.results
                                    if isinstance(r, dict)])

    @property
    def step(self):
        if not any(self.results):
            return 'N/A'
        return int(np.median([int(r['step']) for r in self.results
                              if isinstance(r, dict)]))

    def print_out(self, verbosity=1):
        to_print = {}
        keys = set('argv label locations'.split())
        if verbosity >= 1:
            self._parse_logs()
            self._running_programs()
            keys = keys.union('status dead'.split())
        if verbosity >= 2:
            keys = keys.union(self.metadata.keys())
        for key in keys:
            to_print[key] = getattr(self, key)
        for key in 'dead'.split():
            if key in to_print and not to_print[key]:
                del to_print[key]
        print(yaml.safe_dump(to_print))

@click.command()
@click.option('--verbosity', '-v', default=1)
def get_status(verbosity):
    for state in get_states():
        if not isinstance(state, dict) or 'state' not in state:
            continue # legacy formats
        if state['state'] == 'alive':
            results = Results(state)
            results.print_out(verbosity)

if __name__ == '__main__':
    get_status()
