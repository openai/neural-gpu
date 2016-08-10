#!/usr/bin/python
from __future__ import print_function
import fileinput

import sys
import numpy as np
import pandas as pd
import argparse
import glob
import scipy.signal
import os
import yaml
import shutil
import joblib
import functools
import re

import collections
import pylab

pylab.rcParams['axes.prop_cycle'] = ("cycler('color', ['b','g','r','c','m','y','k'] + "
                                     "['orange', 'darkgreen', 'indigo', 'gold', 'fuchsia'])")
#pylab.rcParams['axes.prop_cycle'] = ("cycler('color', 'bgrcmyk'*2)")

parser = argparse.ArgumentParser(description='Get scores')

RESULT='score'


parser.add_argument("--key", type=str, default="len,score,errors")
parser.add_argument("--job", type=str, default='plot')
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--exclude_opts", type=str, default=None)
parser.add_argument("--title", type=str, default='')
parser.add_argument("--savedir", type=str, default='')
parser.add_argument("--min-length", type=int, default=2)
parser.add_argument("--dirs-in-name", type=int, default=2)
parser.add_argument("--one-legend", type=bool, default=True)
parser.add_argument("--skip-dir", action='store_true')
parser.add_argument("--success", action='store_true')
parser.add_argument("--recache", action='store_true')
parser.add_argument("--median", action='store_true')
parser.add_argument("--smoothing", type=int, default='1')
parser.add_argument('files', type=str, nargs='+',
                    help='Log files to examine')

memory = joblib.Memory(cachedir='/home/ecprice/neural_gpu/cache',
                       verbose=1)

def recache(f):
    g = memory.cache(f)
    @functools.wraps(g)
    def cached(*args, **kwargs):
        if recache.do_recache:
            try:
                shutil.rmtree(g.get_output_dir(*args, **kwargs)[0])
            except OSError: # Not actually in cache
                pass
        return g(*args, **kwargs)
    return cached

recache.do_recache = False

@recache
def get_results_dict(fname):
    if not os.path.exists(fname):
        return {}
    answer = {}
    with open(fname) as f:
        for line in f:
            words = line.split()
            if not words: # Blank line on restart
                continue
            loc, val = words[:2]
            taskname = words[2]
            if taskname not in answer:
                answer[taskname] = pd.Series(name=RESULT)
            try:
                answer[taskname].loc[int(loc)] = float(val)
            except ValueError:
                pass
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

@recache
def get_dfs(dirname, tasknames):
    fname = dirname+'/steps'
    if not os.path.exists(fname):
        fname = dirname+'/log0'
    data_series = {t:{} for t in tasknames}
    for d in get_scores_dict(fname):
        lens = d['len'].split('/')
        if 'progressive_curriculum=5' in lens:
            missing = [i for i in range(len(tasknames)) if lens[i] != '41'] or [len(tasknames)-1]
        else:
            missing = []
        for key in d:
            vals = d[key].split('/')
            if len(vals) == 1:
                vals *= len(tasknames)
            elif len(vals) == len(missing):
                vals2 = [np.nan]*len(tasknames)
                for i, v in zip(missing, vals):
                    vals2[i] = v
                vals = vals2
            elif len(vals) < len(tasknames): #Failed to get data for one
                vals = [np.nan]*len(tasknames)
            for val, task in zip(vals, tasknames):
                data_series[task].setdefault(key, []).append(float(val))
    dfs = {}
    for task in data_series:
        try:
            dfs[task] = pd.DataFrame(data_series[task], index=data_series[task]['step'])
            dfs[task] = dfs[task].drop_duplicates(subset='step', keep='last')
        except KeyError: #Hasn't gotten to 'step' line yet
            pass
    return dfs

def matches(fname, exclude_opts):
    if exclude_opts:
        for opt in exclude_opts.split('|'):
            if opt in fname:
                return True
    return False

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

    def last_loc(self):
        options = ([d.index[-1] for d in self.result_dfs.values()] +
                   [d.index[-1] for d in self.dfs.values()])
        return max(options or [3])

    def get_scores(self, key, task):
        if key == RESULT:
            self._load_results()
            if task is None:
                assert len(self.result_dfs) == 1
                task = self.result_dfs.keys()[0]
            if task not in self.result_dfs:
                basic = pd.Series([1], name=RESULT)
                basic.loc[self.last_loc()] = 1
                return basic
            return self.result_dfs[task]
        else:
            self._load_scores()
            if task is None:
                assert len(self.dfs) == 1
                task = self.dfs.keys()[0]
            if task not in self.dfs:
                return None
            return self.dfs[task].get(key)

    def _load_results(self):
        if self.result_dfs:
            return
        self.result_dfs = get_results_dict(self.dirname+'/results')

    def _load_scores(self):
        if self.dfs:
            return
        self.dfs = get_dfs(self.dirname, self.tasknames)

    def commandline(self):
        return open(os.path.join(self.dirname, 'commandline')).read().split()

    def total_steps(self):
        lens = self.get_scores('len', self.tasknames[0])
        return lens.index[-1].item() if lens is not None else None

def get_name(fname):
    fname = remove_defaults(fname)
    return '/'.join(fname.split('/')[:2])

def plot_start(key):
    pylab.xlabel('Steps of training')
    if key:
        pylab.ylabel(key)
    else:
        pylab.ylabel('Sequence error on large input')

def plot_results(fname, frame):
    label = get_name(fname)#fname
    if frame is None: #Just put in legend
        pylab.plot([], label=label, marker='o')
        return
    x = frame.index
    ysets = list(frame.T.values)
    if args.smoothing > 1:
        f = lambda y: scipy.signal.savgol_filter(y, args.smoothing, 1) if len(y) > args.smoothing else y
    else:
        f = lambda y: y
    ysets = np.array(map(f, ysets)).T
    y = np.median(ysets, axis=1) if args.median else ysets.mean(axis=1)
    v=pylab.plot(x, y,
               label=label,
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

def remove_defaults(fname):
    for default in ['max_steps=200000',
                    'forward_max=201',
#                    'forward_max=401',
                    'max_length=41',
                    'do_resnet=False',
                    'do_binarization=0.0',
                    'do_batchnorm=0',
                    'do_shifter=0',
                    'cutoff_tanh=0.0',
                    'input_height=2',
                    ]:
        fname = fname.replace(default+'-', '')
        if fname.endswith(default):
            fname = fname[:-len(default)-1]
    fname = fname.replace('badde,baddet', 'badde')
    fname = fname.replace('baddet,badde', 'baddet')
    fname = re.sub('(task=[^-]*)-(nmaps=[0-9]*)', r'\2-\1', fname)
    return fname

def get_key(fname):
    fname = fname.split('-seed')[0]
    fname = '/'.join(fname.split('/')[-args.dirs_in_name:])
    fname = remove_defaults(fname)
    return fname

def get_prefix(fileset):
    longest_cp = os.path.commonprefix(fileset)
    i = 1
    while i <= len(longest_cp) and longest_cp[-i] not in '-/':
        i += 1
    return longest_cp[:len(longest_cp)+ 1-i]

badkeys = set()
def plot_all(func, scores, column=None, taskset=None):
    d = {}
    for s in scores:
        d.setdefault(s.key, []).append(s)

    for key in sorted(d):
        if matches(key, args.exclude_opts):
            continue
        for task in d[key][0].tasknames:
            if (key, task) in badkeys:
                continue
            if task not in taskset:
                continue
            columns = [score.get_scores(column, task)
                       for score in d[key]]
            def strip_last(c):
                if c is None or c.index[-1] != 200200:
                    return c
                return c[c.index[:-1]]
            columns = map(strip_last, columns)
            if column == 'len' and args.success:
                if not [c for c in columns if c is not None and c.values[-1] > 10]:
                    badkeys.add((key, task))
                    continue
            median_len = np.median([len(c) for c in columns if c is not None])
            data = pd.DataFrame([c for c in columns if c is not None and len(c) >= median_len / 2 and len(c) >= args.min_length]).T
            if not len(data):
                func(score.args_str(), None)
                continue
            data.loc[data.first_valid_index()] = data.loc[data.first_valid_index()].fillna(1)
            data = data.interpolate(method='nearest')
            func(score.args_str(), data)

legend_locs = dict(score='upper right',
                   len='lower right',
                   errors='upper right')

def get_filter(column):
    if column == 'len':
        return lambda x: x == 41
    else:
        return lambda x: x < 0.01

def get_print_results(scores, column, avg=5):
    assert len(set(x.key for x in scores)) == 1
    ans = {}
    for task in scores[0].tasknames:
        columns = [score.get_scores(column, task) for score in scores]
        columns = [c for c in columns if c is not None]
        if not columns:
            continue
        last_values = [np.mean(c.values[-avg:]).item() for c in columns]
        filt = get_filter(column)
        times = [c.index[np.where(filt(c))] for c in columns]
        first_time = [t[0].item() if len(t) else None for t in times]
        ans[task] = {}
        ans[task]['last'] = last_values
        ans[task]['first-time'] = first_time
        ans[task]['fraction'] = len([x for x in first_time if x is not None]) * 1. / len(times)

    return ans

def construct_parsed_data(scores, columns, save_dir):
    d = {}
    for s in scores:
        d.setdefault(s.key, []).append(s)

    for i, key in enumerate(d):
        ans = {}
        ans['metadata'] = dict(commandline=d[key][0].commandline(),
                               count = len(d[key]),
                               steps = [s.total_steps() for s in d[key]]
        )
        for col in columns:
            ans[col] = get_print_results(d[key], col)
        with open(os.path.join(save_dir, key), 'w') as f:
            print(yaml.safe_dump(ans), file=f)
        print("Done %s/%s" % (i+1, len(d)))

@recache
def is_valid_dir(f):
    return os.path.exists(os.path.join(f, 'log0'))

if __name__ == '__main__':
    args =  parser.parse_args()
    recache.do_recache = args.recache
    print("Started")
    all_tasks = sorted(set(x for file in args.files for x in get_tasks(get_key(file))))
    if args.task:
        all_tasks = args.task.split(',')
    keys = args.key.split(',')
    prefix = get_prefix(args.files)
    scores = [Scores(f, prefix=prefix) for f in args.files if is_valid_dir(f)]
    if args.job == 'parse':
        if args.savedir:
            construct_parsed_data(scores, keys, args.savedir)
        else:
            ans = {}
            for key in keys:
                ans[key] = get_print_results(scores, key)
            print(yaml.safe_dump(ans))
    elif args.job == 'plot':
        title = args.title
        if not title:
            title = os.path.split(args.files[0])[-2]
        title += '\nCommon args: %s' % prefix
        pylab.suptitle(title)
        for ki, key in enumerate(keys):
            for i, task in enumerate(all_tasks):
                plot_index = ki*len(all_tasks) + i+1
                print("Subplot %s/%s" % (plot_index, len(all_tasks)*len(keys)))
                pylab.subplot(len(keys), len(all_tasks), plot_index)
                plot_start(key)
                plot_all(plot_results, scores, column=key, taskset = [task])
                if not args.one_legend or (ki == len(keys)-1 and
                                           (i == len(all_tasks)-1 or 1)):
                    pylab.legend(loc=legend_locs.get(key, 0))
                pylab.title('Task %s' % task)
                pylab.ylim((0, None))
                pylab.xlim((0,None))
        #pylab.tight_layout()
        pylab.show()
