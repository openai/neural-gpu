#!/usr/bin/python3
from __future__ import print_function

import datetime
import glob
import subprocess
import collections
import sys
import threading
import time
import os

import yaml
import numpy as np

import click
import cirrascale.client

def my_gpus(user='ecprice'):
    d = cirrascale.client.get_gpu_status()
    gpus = set(g for k in d for g in d[k] for p in g.processes
                if p.username == user)
    ans = {}
    for g in gpus:
        ans.setdefault(g.host, set()).add(g.index)
    return ans

def other_gpus(results):
    used_gpus = my_gpus()
    known_gpus = {}
    for r in results:
        for k in r.locations:
            known_gpus.setdefault(k, set()).update(r.locations[k])
    missing = {}
    for k in used_gpus:
        here = used_gpus[k] - known_gpus.get(k, set())
        if here:
            missing[k] = sorted(list(here))
    return missing

def get_fnames():
    return glob.glob('servers/*')

def get_states():
    for fname in sorted(get_fnames()):
        with open(fname) as f:
            state = yaml.load(f)
            if not isinstance(state, dict) or 'state' not in state:
                continue # legacy formats
            yield state

def load_log(dir):
    fname = '%s/steps' % dir
    if not os.path.exists(fname):
        fname = '%s/log0' % dir
    try:
        lines = open(fname).readlines()
    except IOError:
        return None
    step_lines = [x for x in lines if x.startswith('step ')]
    if not step_lines:
        fname = '%s/log0' % dir
        try:
            lines = open(fname).readlines()
        except IOError:
            return None
        return lines[-1].strip()
    else:
        data = step_lines[-1].split()
        mapping = collections.OrderedDict(zip(data[::2], data[1::2]))
        mapping['last_update'] = os.stat(fname).st_mtime
        return mapping

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
        output = subprocess.check_output(['ssh', server, cmd])
    return output.split()

class Results(object):
    def __init__(self, metadata):
        self.__dict__.update(metadata)
        self.metadata = metadata

    @property
    def age(self):
        start_time = self.metadata.get('date')
        if start_time is None:
            return 'old'
        else:
            return str(datetime.datetime.now() - start_time)

    @property
    def last_update(self):
        vals = [res for res in self.results if isinstance(res, dict)]
        last_t = max([v['last_update'] for v in vals])
        return str(datetime.timedelta(seconds=time.time() - last_t))

    def _parse_logs(self):
        if hasattr(self, 'results'):
            return
        dirs = glob.glob('cachedlogs/%s/*' % self.label)
        self.results = [load_log(dir) for dir in dirs]

    def _running_programs(self):
        if hasattr(self, 'dead'):
            return
        self.dead = {}
        for server in self.locations:
            still_running = locked_on_server(server)
            dead = [x for x in self.locations[server] if x not in still_running]
            if dead:
                self.dead[server] = dead

    def __repr__(self):
        return '<Results object for %s>' % self.label

    @property
    def status(self):
        vals = [res for res in self.results if isinstance(res, dict)]
        if vals:
            answer = collections.OrderedDict()
            for key in vals[0]:
                if key == 'last_update':
                    continue
                table = [list(map(float, v.get(key, 'nan').split('/'))) for v in vals]
                min_length = min(map(len, table))
                table = [row[:min_length] for row in table]
                answer[key] = np.min(table, axis=0)
            formatting = dict(step='%d', len='%d')
            def fmt_val(val, key):
                if isinstance(val, collections.Iterable):
                    return '/'.join([fmt_val(v, key) for v in val])
                return formatting.get(key, '%0.3g') % val
            return ' '.join('%s %s' % (key, fmt_val(v, key)) for key, v in answer.items())
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
        keys = set('argv age last_update label locations'.split())
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


class RunFunc(threading.Thread):
    def __init__(self, f):
        threading.Thread.__init__(self)
        self.f = f

    def run(self):
        self.f()

@click.command()
@click.option('--verbosity', '-v', default=1)
def get_status(verbosity):
    results = [Results(state) for state in get_states()
               if state['state'] == 'alive']
    if verbosity >= 1:
        threads = []
        for result in results:
            threads.append(RunFunc(result._parse_logs))
            threads.append(RunFunc(result._running_programs))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    for result in results:
        result.print_out(verbosity)

    missing_gpus = other_gpus(results)
    if missing_gpus:
        print(yaml.safe_dump(dict(other=missing_gpus)))

if __name__ == '__main__':
    get_status()
