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
from matplotlib import rc
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

rc('font',  size='12')
rc('text', usetex=True)
rc('axes', labelsize='large')

rc('axes', prop_cycle="cycler('color', ['b','g','r','c','m','y','k'] + "
   "['orange', 'darkgreen', 'indigo', 'gold', 'fuchsia'])")
#pylab.rcParams['axes.prop_cycle'] = ("cycler('color', 'bgrcmyk'*2)")

parser = argparse.ArgumentParser(description='Get scores')

RESULT='score'


parser.add_argument("--key", type=str, default="seq-errors,score")
parser.add_argument("--job", type=str, default='plot')
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--exclude_opts", type=str, default=None)
parser.add_argument("--title", type=str, default='')
parser.add_argument("--titles", type=str, default='')
parser.add_argument("--savedir", type=str, default='')
parser.add_argument("--min-length", type=int, default=2)
parser.add_argument("--dirs-in-name", type=int, default=1)
parser.add_argument("--one-legend", type=bool, default=True)
parser.add_argument("--global-legend", type=str, default='')
parser.add_argument("--save-to", type=str, default='')
parser.add_argument("--skip-dir", action='store_true')
parser.add_argument("--success", action='store_true')
parser.add_argument("--recache", action='store_true')
parser.add_argument("--separate_seeds", action='store_true')
parser.add_argument("--median", action='store_true')
parser.add_argument("--order", type=str, default='')
parser.add_argument("--colorcycle", type=str, default='')
parser.add_argument("--std", type=bool, default=True)
parser.add_argument("--smoothing", type=int, default='11')
parser.add_argument("--remove_strings", type=str, default='')
parser.add_argument("--remove_strings2", type=str, default='')
parser.add_argument('files', type=str, nargs='+',
                    help='Log files to examine')
parser.add_argument("--xlims", type=str, default='')
parser.add_argument("--ylims", type=str, default='')

parser.add_argument("--nbinsx", type=str, default='')
parser.add_argument("--nbinsy", type=str, default='')

parser.add_argument("--xticks", type=str, default='')
parser.add_argument("--yticks", type=str, default='')
parser.add_argument("--lw", type=int, default=3)

parser.add_argument('--traces', dest='traces', action='store_true')
parser.add_argument('--no-traces', dest='traces', action='store_false')
parser.set_defaults(traces=False)

parser.add_argument('--simplify', dest='simplify', action='store_true')
parser.add_argument('--no-simplify', dest='simplify', action='store_false')
parser.set_defaults(simplify=True)

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
        if 'progressive_curriculum=5' in fname:
            missing = [i for i in range(len(tasknames)) if lens[i] != '41'] or [len(tasknames)-1]
        else:
            missing = []
        for key in d:
            vals = d[key].split('/')
            if len(vals) == 1 and (key == 'step' or not missing):
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
        ans = max(options or [3])
        if ans == 200200:
            ans -= 200
        return ans

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
            if key in ['errors', 'seq-errors']:
                scale = 0.01
            else:
                scale = 1
            return self.dfs[task].get(key) * scale

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
    for s in args.remove_strings2.split('|'):
        fname = fname.replace(s, '')
    ans = '/'.join(fname.split('/')[:2])
    ans = ans.replace('_', r'\_')
    return ans

def plot_startx(key):
    pylab.xlabel('Steps of training')
def plot_starty(key):
    if key:
        mapping = {'score': 'Test error',
                   'seq-errors': 'Training error',}
        pylab.ylabel(mapping.get(key, key))
    else:
        pylab.ylabel('Sequence error on large input')

def plot_results(fname, frame):
    label = get_name(fname)#fname
    fmt = dict()
    if frame is None: #Just put in legend
        pylab.plot([], label=label, **fmt)
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
               **fmt
    )
    if args.traces:
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
                    'max_steps=40000',
                    'forward_max=201',
#                    'forward_max=401',
                    'max_length=41',
                    'do_resnet=False',
                    'do_binarization=0.0',
                    'do_batchnorm=0',
                    'do_shifter=0',
                    'cutoff_tanh=0.0',
                    'input_height=2',
                    'batch_size=32',
                    ]:
        fname = fname.replace(default+'-', '')
        if fname.endswith(default):
            fname = fname[:-len(default)-1]
    if args.simplify:
        fname = fname.replace('badde,baddet', 'badde')
        fname = fname.replace('baddet,badde', 'baddet')
    fname = re.sub('(task=[^-]*)-(nmaps=[0-9]*)', r'\2-\1', fname)
    for s in args.remove_strings.split('|'):
        fname = fname.replace(s, '')
    return fname

def get_key(fname):
    if not args.separate_seeds:
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
def plot_all(func, scores, column=None, taskset=None, order=None):
    d = {}
    for s in scores:
        d.setdefault(s.key, []).append(s)

    keys = sorted(d)
    ordered_keys = []
    for key in keys:
        if matches(key, args.exclude_opts):
            continue
        ordered_keys.append(key)
    if order:
        ordered_keys = [ordered_keys[i-1] for i in order]
    for key in ordered_keys:
        for task in d[key][0].tasknames:
            if (key, task) in badkeys:
                continue
            if task not in taskset:
                continue
            columns = [score.get_scores(column, task)
                       for score in d[key]]
            columns = [c for c in columns if c is not None and not c.isnull().all()]
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
            if column != 'score':
                columns = [c for c in columns if len(c) >= median_len / 2 and len(c) >= args.min_length]
            else:
                length_fn = lambda c: c.last_valid_index() // 200
                columns = [c for c in columns if length_fn(c) >= median_len / 2 and length_fn(c) >= args.min_length and len(c) > 1]
            data = pd.DataFrame(columns).T
            if not len(data):
                func(score.args_str(), None)
                continue
            try:
                loc = data.first_valid_index()
            except IndexError:
                continue
            data.loc[loc] = data.loc[loc].fillna(1)
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

def get_print_results(scores, column, avg=10):
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
        if s.total_steps() < 50000:
            continue
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

def get_value(s, i):
    v = s.split('|')
    if len(v) == 1:
        return v[0]
    return v[i]

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
        if args.colorcycle:
            if ',' in args.colorcycle:
                lst = args.colorcycle.split(',')
            else:
                lst = list(args.colorcycle)
            rc('axes', prop_cycle=matplotlib.cycler('color', lst))

        rc('lines', linewidth=args.lw)
        title = args.title
        if not title:
            title = os.path.split(args.files[0])[-2]
            title += '\nCommon args: %s' % prefix
        pylab.suptitle(title, size=18)
        goal_xlim = None
        axes = [[None for _ in range(len(all_tasks))] for _ in range(len(keys))]

        xlims = [map(float, c.split(',')) for c in args.xlims.split(';') if c]
        ylims = [map(float, c.split(',')) for c in args.ylims.split(';') if c]
        fig = pylab.figure(1)
        gs = gridspec.GridSpec(len(keys), len(all_tasks))
        for ki, key in enumerate(keys):
            for i, task in enumerate(all_tasks):
                plot_index = ki*len(all_tasks) + i
                print("Subplot %s/%s" % (plot_index+1, len(all_tasks)*len(keys)))
                sharex = axes[0][i]
                axes[ki][i] = fig.add_subplot(gs[plot_index], sharex=sharex)
                if ki == len(keys)-1:
                    plot_startx(key)
                if i == 0:
                    plot_starty(key)
                order = get_value(args.order, i)
                if order:
                    order = map(int, order.split(','))
                plot_all(plot_results, scores, column=key, taskset = [task], order=order)
                if not args.global_legend and (not args.one_legend or (ki == len(keys)-1 and
                                           (i == len(all_tasks)-1 or 1))):
                    pylab.legend(loc=legend_locs.get(key, 0))
                if not args.titles:
                    pylab.title('Task %s' % task)
                else:
                    pylab.title(args.titles.split('|')[plot_index])
                maxy = None
                if key in ('score', 'errors', 'seq-errors'):
                    maxy = 1
                    axes[ki][i].yaxis.set_major_formatter(mtick.FuncFormatter(
                        lambda x, pos: '% 2d%%' % (x*100)
                    ))
                if ylims:
                    pylab.ylim(ylims[ki])
                else:
                    pylab.ylim((0, maxy))
                if xlims:
                    pylab.xlim(xlims[i])
                else:
                    pylab.xlim((0, None))

                if args.nbinsx:
                    pylab.locator_params(axis='x',nbins=int(get_value(args.nbinsx, i)))
                if args.nbinsy:
                    pylab.locator_params(axis='y',nbins=int(get_value(args.nbinsy, ki)))
                if args.yticks:
                    pylab.yticks(map(float, get_value(args.yticks, ki).split(',')))

                axes[ki][i].xaxis.set_major_formatter(mtick.FuncFormatter(
                    lambda x, pos: '%dk' % (x//1000) if x else '0'
                ))
        #import ipdb;ipdb.set_trace()
        rect = [0,0,1,.92]
        if args.global_legend:
            lines,labels = axes[0][0].get_legend_handles_labels()
            my_labels = args.global_legend.split('|')
            if my_labels == ['1']:
                my_labels = labels
            if my_labels != ['0']:
                fig.legend(lines, my_labels,
                           loc='lower center', ncol=2, labelspacing=0.)
                rect = [0, 0.1, 1, 0.92]
        gs.tight_layout(fig, rect=rect)
        if args.save_to:
            pylab.savefig(args.save_to)
        else:
            pylab.show()
