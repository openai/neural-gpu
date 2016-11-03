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
import pickle

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
parser.add_argument("--only-same", type=bool, default=False)
parser.add_argument("--smoothing", type=int, default='11')
parser.add_argument("--remove_strings", type=str, default='')
parser.add_argument("--remove_strings2", type=str, default='')
parser.add_argument('files', type=str, nargs='+',
                    help='Log files to examine')
parser.add_argument("--xlims", type=str, default='')
parser.add_argument("--ylims", type=str, default='')

parser.add_argument("--nbinsx", type=str, default='')
parser.add_argument("--nbinsy", type=str, default='')

parser.add_argument("--overlay", type=int, default=1)
parser.add_argument("--only-plot", type=str, default=None)

parser.add_argument("--xticks", type=str, default='')
parser.add_argument("--yticks", type=str, default='')
parser.add_argument("--lw", type=int, default=3)
parser.add_argument("--figsize", type=str, default='')

parser.add_argument('--traces', dest='traces', action='store_true')
parser.add_argument('--no-traces', dest='traces', action='store_false')
parser.set_defaults(traces=False)

parser.add_argument('--startx', dest='startx', action='store_true')
parser.add_argument('--no-startx', dest='startx', action='store_false')
parser.set_defaults(startx=True)
parser.add_argument('--starty', dest='starty', action='store_true')
parser.add_argument('--no-starty', dest='starty', action='store_false')
parser.set_defaults(starty=True)

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
        if ans == 200200 or ans == 60200:
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
                    'max_steps=60000',
                    'max_steps=80000',
                    'max_steps=100000',
                    'forward_max=201',
#                    'forward_max=401',
                    'max_length=41',
                    'time_till_eval=4',
                    'always_large=True',
                    'do_resnet=False',
                    'do_binarization=0.0',
                    'do_batchnorm=0',
                    'do_shifter=0',
                    'progressive_curriculum=False',
                    'cutoff_tanh=0.0',
                    'input_height=2',
                    'batch_size=32',
                    ]:
        fname = fname.replace(default+'-', '')
        if fname.endswith(default):
            fname = fname[:-len(default)-1]
    if fname.startswith('random_seed='):
        fname = fname.split('-', 1)[1]
    if 'task' in fname and len(fname.split('task=')[1].split('-')[0].split(',')) == 1:
        for s in ['2', '3', '4', '5', 'True']:
            fname = fname.replace('-progressive_curriculum=%s' % s, '')
    if args.simplify:
        fname = fname.replace('badd,baddt', 'badd')
        fname = fname.replace('baddt,badd', 'baddt')
        fname = fname.replace('badde,baddet', 'badde')
        fname = fname.replace('baddet,badde', 'baddet')
        fname = fname.replace('baddz,baddzt', 'baddz')
        fname = fname.replace('baddzt,baddz', 'baddzt')
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

def sort_key_fn(label):
    return label.replace('nmaps=24', 'nmaps=024')

badkeys = set()
def plot_all(func, scores, column=None, taskset=None, order=None):
    d = {}
    for s in scores:
        d.setdefault(s.key, []).append(s)

    keys = sorted(d, key=sort_key_fn)
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
                median_len = np.median(map(length_fn, columns))
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

gs = None

def run_plots(args, scores, all_tasks, keys):
    global gs
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
    pylab.suptitle(title, size=18)
    goal_xlim = None
    axes = [[None for _ in range(len(all_tasks))] for _ in range(len(keys))]

    figkws = {}
    if args.figsize:
        figkws['figsize']=map(int, args.figsize.split(','))
    fig = pylab.figure(1,**figkws)
    task_overlays = args.overlay
    if gs is None:
        gs = gridspec.GridSpec(len(keys), len(all_tasks) / task_overlays)
    for ki, key in enumerate(keys):
        for i, task in enumerate(all_tasks):
            full_plot_index = ki*len(all_tasks) + i
            plot_index = full_plot_index // task_overlays
            if args.only_plot and plot_index + 1 != int(args.only_plot.split(',')[0]):
                continue
            print("Subplot %s/%s" % (full_plot_index+1, len(all_tasks)*len(keys)))
            sharex = axes[0][i]
            if args.only_plot:
                newloc = int(args.only_plot.split(',')[1])
                ax = fig.add_subplot(gs[newloc-1])
                axes[ki][i] = ax
            else:
                axes[ki][i] = fig.add_subplot(gs[plot_index], sharex=sharex)
            if ki == len(keys)-1 and args.startx:
                plot_startx(key)
            if i == 0 and args.starty:
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
                    lambda x, pos: '% 2d\\%%' % (x*100)
                ))
            ylims = map(float, get_value(args.ylims, ki).split(',')) if args.ylims else (0,1)
            pylab.ylim(ylims)
            xlims = map(float, get_value(args.xlims, i).split(',')) if args.xlims else (0,None)
            pylab.xlim(xlims)

            if args.nbinsx:
                pylab.locator_params(axis='x',nbins=int(get_value(args.nbinsx, i)))
            if args.nbinsy:
                pylab.locator_params(axis='y',nbins=int(get_value(args.nbinsy, ki)))
            if args.yticks:
                pylab.yticks(map(float, get_value(args.yticks, ki).split(',')))

            axes[ki][i].xaxis.set_major_formatter(mtick.FuncFormatter(
                lambda x, pos: '%dk' % (x//1000) if x else '0'
            ))
    rect = [0,0,1,.92]
    if args.global_legend:
        if not args.only_plot:
            ax = [row for row in axes if row[0]][0][0]
        lines,labels = ax.get_legend_handles_labels()
        my_labels = args.global_legend.split('|')
        if my_labels == ['1']:
            my_labels = labels
        if my_labels != ['0']:
            if my_labels != ['2']:
                fig.legend(lines, my_labels, loc='lower center',
                           ncol=2, labelspacing=0.)
            rect = [0, 0.1, 1, 0.92]
    gs.tight_layout(fig, rect=rect)
    if args.save_to:
        pylab.savefig(args.save_to)
    else:
        pylab.show()

def get_value(s, i):
    v = s.split('|')
    if len(v) == 1:
        return v[0]
    return v[i]


def main():
    global args
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
        run_plots(args, scores, all_tasks, keys)

'''
python  get_pretty_score.py cachedlogs/{Jul,A}*/*24*={b,}add{e,z,}*  --task badd,badde,baddz,add,adde,addz --remove_strings '|-progressive_|curriculum=2|curriculum=5' --exclude='forward_max|rx_step|cutoff|binar|grad_noise|t,|dropout|badd,add|batchnorm|resnet'  --min-length 30 --title 'Alignment helps addition' --titles='||Binary addition, 24 filters|'  --xlims='0,30000' --nbinsx=3 --global-legend='Padded|Aligned|Unpadded' --overlay=3 --save-to=moo.pdf --no-startx dump magic1.pickle
python get_pretty_score.py cachedlogs/{Jul,A}*/*128*={b,}add{e,z,}*  --task badd,badde,baddz,add,adde,addz --remove_strings '|-progressive_|curriculum=2|curriculum=5' --exclude='kbadd|qbadd|qadd|3badd|3add|kadd|curric|forward_max|rx_step|cutoff|binar|grad_noise|t,|dropout|badd,add|curriculum|resnet|batchn'  --min-length 30 --title 'Alignment helps addition' --titles='Binary, 128 filters|Decimal, 128 filters||'  --xlims='0,30000' --nbinsx=3  --overlay=3 --save-to=moo.pdf --global-legend='Padded|Aligned|Unpadded' dump magic2.pickle

python  get_pretty_score.py cachedlogs/{Jul,A}*/*24*=bmul{e,z,}-*  --task mul,mule,mulz,bmul,bmule,bmulz --remove_strings '|-progressive_|curriculum=2|curriculum=5|max_steps=80000-' --exclude='forward_max|rx_step|cutoff|binar|grad_noise|t,|dropout|batchn|resn|layer'  --min-length 30 --title 'Alignment hurts multiplication'  --overlay=3 --global-legend=2  --titles '|||Binary multiplication, 24 filters' --no-startx --xlims='0,100000' --save-to=moo.pdf dump magic3.pickle

python  get_pretty_score.py cachedlogs/{Jul,A}*/*128*=bmul{e,z,}-*  --task mul,mule,mulz,bmul,bmule,bmulz --remove_strings '|-progressive_|curriculum=2|curriculum=5|max_steps=80000-' --exclude='forward_max|rx_step|cutoff|binar|grad_noise|t,|dropout|batchn|resn|layer'  --min-length 30 --title 'Alignment helps addition, hurts multiplication'  --overlay=3 --global-legend=2  --titles '|||Binary multiplication, 128 filters' --save-to=moo.pdf dump magic4.pickle

'''
if __name__ == '__main__':
    if sys.argv[1] == 'magic':
        for i, loc in enumerate(['3,1', '3,3', '4,2', '4,4']):
            sys.argv[1:] = pickle.load(open('magic%s.pickle' % (i+1))) + ['--only-plot', loc]
            main()
        sys.exit()
    if len(sys.argv) > 1 and 'dump' == sys.argv[-2]:
        loc = sys.argv.pop()
        sys.argv.pop()
        pickle.dump(sys.argv[1:], open(loc, 'w'))
        sys.exit()
    main()
