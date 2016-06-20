#!/usr/bin/python
import fileinput

import sys
import numpy as np
import pandas as pd
import argparse
import glob
import scipy.signal
import os

import collections

parser = argparse.ArgumentParser(description='Get scores')

RESULT='score'

parser.add_argument("--key", type=str, default=RESULT)
parser.add_argument("--task", type=str, default='plot')
parser.add_argument("--title", type=str, default='')
parser.add_argument("--median", action='store_true')
parser.add_argument("--smoothing", type=int, default='1')
parser.add_argument('files', type=str, nargs='+',
                    help='Log files to examine')

def get_results_dict(fname):
    answer = {}
    with open(fname) as f:
        for line in f:
            words = line.split()
            loc, val = words[:2]
            taskname = words[2]
            if taskname not in answer:
                answer[taskname] = pd.Series(name=RESULT)
            answer[taskname].loc[int(loc)] = float(val)
    return answer

def get_scores_dict(fname):
    with open(fname) as f:
        for line in f:
            if line.startswith('step '):
                entries = line.split()
                d = collections.OrderedDict(zip(entries[::2], entries[1::2]))
                try:
                    yield d
                except ValueError:
                    break

class Scores(object):
    def __init__(self, dirname, tasknames=None, prefix=''):
        self.dirname = dirname
        self.index = 0
        if tasknames is None:
            tasknames = get_tasks(self.key)
        self.tasknames = tasknames
        self.prefix = prefix
        self.result_dfs = {}
        self.dfs = {}

    @property
    def key(self):
        return get_key(self.dirname)

    def args_str(self, task=None):
        label = get_key(self.dirname[len(self.prefix):])
        return (label +
                (' (%s)' % task if task and len(self.tasknames) > 1 else ''))

    def get_scores(self, key, task):
        if key == RESULT:
            self._load_results()
            if task is None:
                assert len(self.result_dfs) == 1
                task = self.result_dfs.keys()[0]
            if task not in self.result_dfs:
                return None
            return self.result_dfs[task]
        else:
            self._load_scores()
            if task is None:
                assert len(self.dfs) == 1
                task = self.dfs.keys()[0]
            if task not in self.dfs:
                return None
            return self.dfs[task][key]

    def _load_results(self):
        if self.result_dfs:
            return
        self.result_dfs = get_results_dict(self.dirname+'/results')

    def _load_scores(self):
        if self.dfs:
            return
        data_series = {t:{} for t in self.tasknames}
        for d in get_scores_dict(self.dirname+'/log0'):
            for key in d:
                vals = d[key].split('/')
                if len(vals) == 1:
                    vals *= len(self.tasknames)
                for val, task in zip(vals, self.tasknames):
                    data_series[task].setdefault(key, []).append(float(val))
        for task in data_series:
            self.dfs[task] = pd.DataFrame(data_series[task], index=data_series[task]['step'])

def get_name(fname):
    return '/'.join(fname.split('/')[:2])

def plot_start(key):
    pylab.xlabel('Steps of training')
    if key:
        pylab.ylabel(key)
    else:
        pylab.ylabel('Sequence error on large input')

def plot_results(fname, frame):
    x = frame.index
    ysets = list(frame.T.values)
    if args.smoothing > 1:
        f = lambda y: scipy.signal.savgol_filter(y, args.smoothing, 1) if len(y) > args.smoothing else y
    else:
        f = lambda y: y
    ysets = np.array(map(f, ysets)).T
    y = np.median(ysets, axis=1) if args.median else ysets.mean(axis=1)
    v=pylab.plot(x, y,
               label=get_name(fname),
               marker='o',
    )
    for ys in list(ysets.T):
        pylab.plot(x, ys, alpha=0.2,
                   color=v[0].get_color(),
        )
    pylab.fill_between(frame.index, ysets.min(axis=1), ysets.max(axis=1),
                       alpha=0.15, color=v[0].get_color())

    #for k in frame.columns:
    #    pylab.scatter(frame.index, frame[k].values, alpha=0.15, color=v[0].get_color())

def get_tasks(key):
    if 'task' not in key:
        return ['rev']
    else:
        locs = key.split('=')
        index = [i for i,a in enumerate(locs) if a.endswith('task')][0]+1
        tasks = locs[index].split('-')[0].split(',')
        return tasks

def get_key(fname):
    return '/'.join(('-'.join(fname.split('-')[:-2])).split('/')[-2:])

def get_prefix(fileset):
    longest_cp = os.path.commonprefix(fileset)
    i = 1
    while i <= len(longest_cp) and longest_cp[-i] not in '-/':
        i += 1
    return longest_cp[:len(longest_cp)+ 1-i]

def plot_all(func, scores, column=None, taskset=None):
    d = {}
    for s in scores:
        d.setdefault(s.key, []).append(s)

    for key in sorted(d):
        for task in d[key][0].tasknames:
            if task not in taskset:
                continue
            columns = [score.get_scores(column, task)
                       for score in d[key]]
            data = pd.DataFrame([c for c in columns if c is not None]).T
            if not len(data):
                continue
            data.loc[data.first_valid_index()] = data.loc[data.first_valid_index()].fillna(1)
            data = data.interpolate(method='nearest')
            func(score.args_str(), data)

legend_locs = dict(score='upper right',
                   len='lower right',
                   errors='upper right')

if __name__ == '__main__':
    args =  parser.parse_args()
    all_tasks = sorted(set(x for file in args.files for x in get_tasks(get_key(file))))
    keys = args.key.split(',')
    prefix = get_prefix(args.files)
    scores = [Scores(f, prefix=prefix) for f in args.files]
    if args.task == 'print':
        raise ValueError()
    elif args.task == 'plot':
        global pylab
        import pylab
        title = args.title
        if not title:
            title = os.path.split(args.files[0])[-2]
        title += '\nCommon args: %s' % prefix
        pylab.suptitle(title)
        for ki, key in enumerate(keys):
            for i, task in enumerate(all_tasks):
                pylab.subplot(len(keys), len(all_tasks), ki*len(all_tasks) + i+1)
                plot_start(key)
                plot_all(plot_results, scores, column=key, taskset = [task])
                pylab.legend(loc=legend_locs.get(key, 0))
                pylab.title('Task %s' % task)
                pylab.ylim((0, None))
                pylab.xlim((0,None))
        #pylab.tight_layout()
        pylab.show()
