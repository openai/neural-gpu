from __future__ import print_function
import yaml
import glob
import subprocess
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
        return dict(zip(data[::2], data[1::2]))

def locked_on_server(server):
    cmd = 'python models/neural_gpu/used_gpus.py'
    output = subprocess.check_output(['ssh', server, cmd])
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
        if any(isinstance(res, dict) for res in self.results):
            return dict(accuracy=self.accuracy,
                        step = self.step)
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
